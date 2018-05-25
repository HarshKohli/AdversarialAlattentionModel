# Author: Harsh Kohli
# Date created: 5/23/2018

import tensorflow as tf
from utils.data_processing import get_batch


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

    def train(self, sess, train_data, dev_data, word_to_id_lookup, config):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)
        best_dev_loss = float('inf')
        for iteration_no in range(config['num_iterations']):
            batch_info = get_batch(train_data, iteration_no, word_to_id_lookup, config)
            feed_dict = self.create_feed_dict(batch_info)
            print('here')

    def initialize_word_embeddings(self, sess, embeddings):
        init_op = tf.global_variables_initializer()
        sess.run(init_op, {self.embeddings_placeholder: embeddings})

    def create_feed_dict(self, batch):
        feed_dict = {
            self.paragraphs: batch['paragraphs'],
            self.questions: batch['questions'],
            self.para_labels: batch['para_labels'],
            self.answer_starts: batch['answer_starts'],
            self.answer_ends: batch['answer_ends'],
            self.para_lengths: batch['para_lengths'],
            self.question_lengths: batch['question_lengths']
        }
        return feed_dict
