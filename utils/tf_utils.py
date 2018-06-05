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
        return tf.concat(hidden_states, axis=2), tf.concat(final_states, axis=2)


def highway_maxout(inputs, hidden_size, pool_size, keep_prob):
    layer1 = maxout_layer(inputs, hidden_size, pool_size, keep_prob)
    layer2 = maxout_layer(layer1, hidden_size, pool_size, keep_prob)
    highway = tf.concat([layer1, layer2], -1)
    output = maxout_layer(highway, 1, pool_size, keep_prob)
    output = tf.squeeze(output, -1)
    return output


def maxout_layer(inputs, outputs, pool_size, keep_prob=1.0):
    inputs = tf.nn.dropout(inputs, keep_prob)
    pool = tf.layers.dense(inputs, outputs * pool_size)
    pool = tf.reshape(pool, (-1, tf.shape(inputs)[1], outputs, pool_size))
    output = tf.reduce_max(pool, -1)
    return output
