# Author: Harsh Kohli
# Date created: 5/26/2018

import tensorflow as tf


def bi_lstm(inputs, seq_len, dim, name, keep_prob):
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fw = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bw = tf.contrib.rnn.LSTMCell(num_units=dim)
        hidden_states, final_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fw, cell_bw=lstm_bw,
                                                                      inputs=inputs, sequence_length=seq_len,
                                                                      dtype=tf.float32, scope=name)
        return hidden_states, final_states


def bi_lstm_4D(inputs, seq_len, dim, input_dim, name, keep_prob):
    reshaped_inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[2], input_dim])
    return bi_lstm(reshaped_inputs, tf.reshape(seq_len, [-1]), dim, name, keep_prob)
