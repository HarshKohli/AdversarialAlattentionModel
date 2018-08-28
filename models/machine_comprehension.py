# Author: Harsh Kohli
# Date created: 5/23/2018

import tensorflow as tf
import numpy as np
import json
import eval.squad_eval as squad_eval
from utils.data_processing import get_batch_input, get_training_batch, get_adversarial_batch, get_strongest_adversaries, \
    get_dev_batch, populate_id_to_answer
from utils.tf_utils import bi_lstm
from models.encoders import dynamic_coattention
from models.decoders import dynamic_pointing_decoder


class MCModel():

    def __init__(self, config):
        self.embeddings_placeholder = tf.placeholder(tf.float32, (config['vocab_size'], config['embedding_size']),
                                                     'embeddings_placeholder')
        self.embeddings = tf.Variable(self.embeddings_placeholder, trainable=False, validate_shape=True)
        self.paragraphs = tf.placeholder(tf.int32, (None, None), 'all_paragraphs')
        self.questions = tf.placeholder(tf.int32, (None, None), 'questions')
        self.answer_starts = tf.placeholder(tf.int32, (None), 'answer_starts')
        self.answer_ends = tf.placeholder(tf.int32, (None), 'answer_ends')
        self.para_lengths = tf.placeholder(tf.int32, (None), 'para_lengths')
        self.question_lengths = tf.placeholder(tf.int32, (None), 'question_lengths')
        self.keep_prob = tf.placeholder(tf.float32, [], 'keep_probability')

        hidden_size = config['hidden_size']
        paragraph_embeddings = tf.nn.embedding_lookup(self.embeddings, self.paragraphs)
        question_embeddings = tf.nn.embedding_lookup(self.embeddings, self.questions)
        passage_states, _ = bi_lstm(paragraph_embeddings, self.para_lengths, hidden_size,
                                    'passage_preprocessor', self.keep_prob)
        question_states, _ = bi_lstm(question_embeddings, self.question_lengths, hidden_size,
                                     'question_preprocessor', self.keep_prob)
        self.encoder_output = dynamic_coattention(passage_states, question_states, self.para_lengths, hidden_size,
                                                  self.keep_prob, 'pqattender')
        self.self_attention_output = dynamic_coattention(passage_states, passage_states, self.para_lengths, hidden_size,
                                                         self.keep_prob, 'selfattender')
        decoder_inputs = tf.concat((self.encoder_output, self.self_attention_output), axis=-1)
        self.start_probs, self.end_probs, self.answers = dynamic_pointing_decoder(decoder_inputs,
                                                                                  self.para_lengths, hidden_size,
                                                                                  config['maxout_pool_size'],
                                                                                  config['decoding_iterations'],
                                                                                  self.keep_prob, 'decoder')
        loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_probs, labels=self.answer_starts)
        loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_probs, labels=self.answer_ends)
        self.loss = tf.reduce_sum(loss1 + loss2)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=config['learning_rate']).minimize(self.loss)

    def train(self, sess, train_data, dev_data, word_to_id_lookup, config):
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        best_dev_loss = float('inf')
        dev_iterations = int(len(dev_data) / config['dev_batch_size']) + 1
        for iteration_no in range(config['num_iterations']):
            batch = get_batch_input(train_data, iteration_no, config['batch_size'], True)
            chunks = np.split(np.asarray(batch), 8)
            predicted_starts, predicted_ends = [], []
            for chunk in chunks:
                adversary_batch = get_adversarial_batch(chunk, word_to_id_lookup, config['extra_vectors'])
                feed_dict = self.create_feed_dict(adversary_batch, 1.0)
                predicted_starts_chunk, predicted_ends_chunk = sess.run([self.start_probs, self.end_probs],
                                                                        feed_dict=feed_dict)
                predicted_starts.extend(predicted_starts_chunk)
                predicted_ends.extend(predicted_ends_chunk)
            best_adversaries = get_strongest_adversaries(batch, predicted_starts, predicted_ends)
            training_batch_info = get_training_batch(batch, best_adversaries, word_to_id_lookup,
                                                     config['extra_vectors'])
            feed_dict = self.create_feed_dict(training_batch_info, config['dropout_keep_prob'])
            loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed_dict)
            if iteration_no % config['checkpoint'] == 0:
                dev_loss = self.run_dev_set(sess, dev_data, dev_iterations, word_to_id_lookup, config)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    saver.save(sess, config['save_dir'])
                    print('Saved best model')
            print(loss)

    def run_dev_set(self, sess, dev_data, dev_iters, word_to_id_lookup, config):
        dev_loss = 0
        id_to_answer_map = {}
        for iteration_no in range(dev_iters):
            batch = get_batch_input(dev_data, iteration_no, config['dev_batch_size'], False)
            dev_batch_info = get_dev_batch(batch, word_to_id_lookup, config['extra_vectors'])
            feed_dict = self.create_feed_dict(dev_batch_info)
            loss, answers = sess.run([self.loss, self.answers], feed_dict=feed_dict)
            dev_loss = dev_loss + loss
            if config['task'] == 'squad':
                populate_id_to_answer(answers, dev_batch_info, id_to_answer_map, batch)
        if config['task'] == 'squad':
            with open(config['test_output'], 'w') as outfile:
                json.dump(id_to_answer_map, outfile)
            squad_eval.main([config['test_path'], config['squad_test_output']])
        return dev_loss

    def test(self, sess, test_data, word_to_id_lookup, config):
        saver = tf.train.Saver()
        saver.restore(sess, config['save_dir'])
        print('model restored')
        id_to_answer_map = {}
        test_iterations = int(len(test_data) / config['dev_batch_size']) + 1
        for iteration_no in range(test_iterations):
            print(iteration_no)
            batch = get_batch_input(test_data, iteration_no, config['dev_batch_size'], False)
            dev_batch_info = get_dev_batch(batch, word_to_id_lookup, config['extra_vectors'])
            feed_dict = self.create_feed_dict(dev_batch_info)
            answers = sess.run([self.answers], feed_dict=feed_dict)[0]
            populate_id_to_answer(answers, dev_batch_info, id_to_answer_map, batch)
        return id_to_answer_map

    def initialize_word_embeddings(self, sess, embeddings):
        init_op = tf.global_variables_initializer()
        sess.run(init_op, {self.embeddings_placeholder: embeddings})

    def create_feed_dict(self, batch, keep_prob=1.0):
        feed_dict = {
            self.paragraphs: batch['paragraphs'],
            self.questions: batch['questions'],
            self.answer_starts: batch['answer_starts'],
            self.answer_ends: batch['answer_ends'],
            self.para_lengths: batch['para_lengths'],
            self.question_lengths: batch['question_lengths'],
            self.keep_prob: keep_prob
        }
        return feed_dict
