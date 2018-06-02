# Author: Harsh Kohli
# Date created: 5/23/2018

import tensorflow as tf
from utils.data_processing import get_batch
from utils.tf_utils import bi_lstm, bi_lstm_4D
from models.encoders import dynamic_coattention


class MCModel():

    def __init__(self, config):
        self.embeddings_placeholder = tf.placeholder(tf.float32, (config['vocab_size'], config['embedding_size']),
                                                     'embeddings_placeholder')
        self.embeddings = tf.Variable(self.embeddings_placeholder, trainable=False, validate_shape=True)
        self.paragraphs = tf.placeholder(tf.int32, (None, None, None), 'all_paragraphs')
        self.questions = tf.placeholder(tf.int32, (None, None), 'questions')
        self.para_labels = tf.placeholder(tf.int32, (None), 'answer_paras')
        self.answer_starts = tf.placeholder(tf.int32, (None), 'answer_starts')
        self.answer_ends = tf.placeholder(tf.int32, (None), 'answer_ends')
        self.para_lengths = tf.placeholder(tf.int32, (None, None), 'para_lengths')
        self.question_lengths = tf.placeholder(tf.int32, (None), 'question_lengths')
        self.keep_prob = tf.placeholder(tf.float32, [], 'keep_probability')

        hidden_size = config['hidden_size']
        paragraph_embeddings = tf.nn.embedding_lookup(self.embeddings, self.paragraphs)
        question_embeddings = tf.nn.embedding_lookup(self.embeddings, self.questions)
        passage_states, passage_final_states = bi_lstm_4D(paragraph_embeddings, self.para_lengths, hidden_size,
                                                          config['embedding_size'], 'passage_preprocessor',
                                                          self.keep_prob)
        passage_hidden_states = tf.concat(passage_states, axis=2)
        question_states, question_final_states = bi_lstm(question_embeddings, self.question_lengths, hidden_size,
                                                         'question_preprocessor', self.keep_prob)
        self.question_hidden_states = tf.concat(question_states, axis=2)
        self.encoded_passages = tf.reshape(passage_hidden_states,
                                           [tf.shape(self.paragraphs)[0], tf.shape(self.paragraphs)[1],
                                            tf.shape(self.paragraphs)[2], 2 * hidden_size])
        self.encoder_output = dynamic_coattention(self.encoded_passages, self.question_hidden_states)
        # self.encoded_questions_broadcasted = tf.reshape(self.question_hidden_states,
        #                                                 (tf.shape(self.encoded_passages)[0], tf.shape(self.encoded_passages)[1],
        #                                                          tf.shape(self.questions)[1], 2 * hidden_size))

    def train(self, sess, primary_train_data, dev_data, word_to_id_lookup, config):
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        best_dev_loss = float('inf')
        for iteration_no in range(config['num_iterations']):
            batch_info = get_batch(primary_train_data, iteration_no, word_to_id_lookup, config)
            feed_dict = self.create_feed_dict(batch_info, config['dropout_keep_prob'])
            _ = sess.run([self.encoder_output], feed_dict=feed_dict)
            print('here')

    def initialize_word_embeddings(self, sess, embeddings):
        init_op = tf.global_variables_initializer()
        sess.run(init_op, {self.embeddings_placeholder: embeddings})

    def create_feed_dict(self, batch, keep_prob):
        feed_dict = {
            self.paragraphs: batch['paragraphs'],
            self.questions: batch['questions'],
            self.para_labels: batch['para_labels'],
            self.answer_starts: batch['answer_starts'],
            self.answer_ends: batch['answer_ends'],
            self.para_lengths: batch['para_lengths'],
            self.question_lengths: batch['question_lengths'],
            self.keep_prob: keep_prob
        }
        return feed_dict
