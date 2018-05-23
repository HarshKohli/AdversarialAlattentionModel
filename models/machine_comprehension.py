# Author: Harsh Kohli
# Date created: 5/23/2018

import tensorflow as tf
from utils.data_processing import get_batch


class MCModel():

    def __init__(self, config):
        self.embeddings_placeholder = tf.placeholder(tf.float32, (config['vocab_size'], config['embedding_size']),
                                                     'embeddings')
        self.paragraphs = tf.placeholder(tf.int32, (None, None, None), 'all_paragraphs')
        self.questions = tf.placeholder(tf.int32, (None, None), 'questions')
        self.para_labels = tf.placeholder(tf.int32, (None), 'answer_paras')
        self.answer_starts = tf.placeholder(tf.int32, (None), 'answer_paras')
        self.answer_ends = tf.placeholder(tf.int32, (None), 'answer_paras')
        self.para_lengths = tf.placeholder(tf.int32, (None, None), 'answer_paras')
        self.question_lengths = tf.placeholder(tf.int32, (None), 'answer_paras')

    def train(self, train_data, dev_data, word_to_id_lookup, embeddings, config):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        best_dev_loss = float('inf')
        for iteration_no in range(config['num_iterations']):
            batch_info = get_batch(train_data, iteration_no, word_to_id_lookup, config)
            feed_dict = self.create_feed_dict(batch_info, embeddings)
            print('here')

    def create_feed_dict(self, batch, embeddings):
        feed_dict = {
            self.embeddings_placeholder: embeddings,
            self.paragraphs: batch['paragraphs'],
            self.questions: batch['questions'],
            self.para_labels: batch['para_labels'],
            self.answer_starts: batch['answer_starts'],
            self.answer_ends: batch['answer_ends'],
            self.para_lengths: batch['para_lengths'],
            self.question_lengths: batch['question_lengths']
        }
        return feed_dict
