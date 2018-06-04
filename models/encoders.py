# Author: Harsh Kohli
# Date created: 5/23/2018

import tensorflow as tf
from utils.tf_utils import bi_lstm

def dynamic_coattention(paragraphs, questions, lengths, dim, keep_prob):
    L = tf.einsum('aij,akj->aik', paragraphs, questions)
    Aq = tf.nn.softmax(L, dim=1)
    Ad = tf.nn.softmax(L, dim=2)
    Cq = tf.einsum('aij,aik->akj', paragraphs, Aq)
    Qcq = tf.concat((questions, Cq), axis=2)
    Cd = tf.einsum('aij,aki->akj', Qcq, Ad)
    final_input = tf.concat((paragraphs, Cd), axis=2)
    states, _ = bi_lstm(final_input, lengths, dim, 'final_dcn_encoder', keep_prob)
    return states
