# Author: Harsh Kohli
# Date created: 5/27/2018

import tensorflow as tf

def pointer_network(sequence, dim, sequence_length, labels, name):
    with tf.variable_scope(name):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=dim, memory=sequence, memory_sequence_length=sequence)
        answer_pointer_cell = tf.contrib.rnn.LSTMCell(num_units=dim)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(answer_pointer_cell, attention_mechanism)
        logits,_ = tf.nn.dynamic_rnn(attention_cell, labels, sequence_length=sequence_length)
        return logits
