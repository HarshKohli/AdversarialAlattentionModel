# Author: Harsh Kohli
# Date created: 5/23/2018

import tensorflow as tf

def dynamic_coattention(paragraphs, questions):
    L = tf.einsum('aijk,alk->aijl', paragraphs, questions)
    # Aq = tf.nn.softmax(L, axis=2)
    # Ad = tf.nn.softmax(L, axis=3)
    # Cq = tf.einsum('aijk,aijl->aikl', paragraphs, Aq)

    return L
