# Author: Harsh Kohli
# Date created: 5/27/2018

import tensorflow as tf


def pointer_network(sequence, dim, sequence_length, labels, name):
    with tf.variable_scope(name):
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=dim, memory=sequence,
                                                                   memory_sequence_length=sequence_length)
        answer_pointer_cell = tf.contrib.rnn.LSTMCell(num_units=dim)
        attention_cell = tf.contrib.seq2seq.AttentionWrapper(answer_pointer_cell, attention_mechanism)
        logits, _ = tf.nn.static_rnn(attention_cell, labels, sequence_length=sequence_length)
        tf.contrib.rnn.MultiRNNCell
        return logits

def decode(Hr, para_lengths, dim, cell_init, name):
    with tf.variable_scope(name):
        V = tf.get_variable("V", [2*dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        Wa = tf.get_variable("Wa", [dim, dim], initializer=tf.contrib.layers.xavier_initializer())
        ba = tf.Variable(tf.zeros([1, dim]), name="ba")
        v = tf.Variable(tf.zeros([dim, 1]), name="v")
        c = tf.Variable(tf.zeros([1]), name="c")
        decoding_cell = tf.contrib.rnn.LSTMCell(num_units=dim)

        hk = cell_init
        cell_state = (hk, hk)
        temp1 = tf.matmul(Hr, V)
        temp2 = tf.matmul(hk, Wa) + ba
        Fk = tf.nn.tanh(temp1 + temp2)
        Beta1 = tf.nn.softmax(tf.matmul(Fk, v) + c)
        lstm_input = tf.matmul(Hr, Beta1)
        hk, cell_state = tf.nn.dynamic_rnn(decoding_cell, lstm_input, sequence_length=para_lengths, initial_state=cell_state)

        temp3 = tf.matmul(Hr, V)
        temp4 = tf.matmul(hk, Wa) + ba
        Fk = tf.nn.tanh(temp3 + temp4)
        Beta2 = tf.nn.softmax(tf.matmul(Fk, v) + c)
        return tuple(Beta1, Beta2)
