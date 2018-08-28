# Author: Harsh Kohli
# Date created: 5/27/2018

import tensorflow as tf
from utils.tf_utils import highway_maxout
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _maybe_mask_score

def dynamic_pointing_decoder(encoding, document_length, dim, pool_size, max_iter, keep_prob, name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        batch_size = tf.shape(encoding)[0]
        lstm_dec = tf.contrib.rnn.LSTMCell(num_units=dim)
        lstm_dec = tf.contrib.rnn.DropoutWrapper(lstm_dec, input_keep_prob=keep_prob)
        start = tf.zeros((batch_size,), dtype=tf.int32)
        end = document_length - 1
        answer = tf.stack([start, end], axis=1)
        state = lstm_dec.zero_state(batch_size, dtype=tf.float32)
        logits = tf.TensorArray(tf.float32, size=max_iter, clear_after_read=False)
        for i in range(max_iter):
            output, state = lstm_dec(start_and_end_encoding(encoding, answer), state)
            logit = decoder_body(encoding, output, answer, dim, pool_size, document_length, keep_prob)
            start_logit, end_logit = logit[:, :, 0], logit[:, :, 1]
            start = tf.argmax(start_logit, axis=1, output_type=tf.int32)
            end = tf.argmax(end_logit, axis=1, output_type=tf.int32)
            answer = tf.stack([start, end], axis=1)
            logits = logits.write(i, logit)
    return logits, start_logit, end_logit, answer

def dcn_loss(logits, answer_span, max_iter):
    logits = logits.concat()
    answer_span_repeated = tf.tile(answer_span, (max_iter, 1))
    start_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:, :, 0], labels=answer_span_repeated[:, 0], name='start_loss')
    end_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:, :, 1], labels=answer_span_repeated[:, 1], name='end_loss')
    start_loss = tf.stack(tf.split(start_loss, max_iter), axis=1)
    end_loss = tf.stack(tf.split(end_loss, max_iter), axis=1)
    loss_per_example = tf.reduce_mean(start_loss + end_loss, axis=1)
    loss = tf.reduce_mean(loss_per_example)
    return loss


def start_and_end_encoding(encoding, answer):
    batch_size = tf.shape(encoding)[0]
    start, end = answer[:, 0], answer[:, 1]
    encoding_start = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), start], axis=1))
    encoding_end = tf.gather_nd(encoding, tf.stack([tf.range(batch_size), end], axis=1))
    return tf.concat([encoding_start, encoding_end], axis=1)


def decoder_body(encoding, state, answer, state_size, pool_size, document_length, keep_prob):
    max_len = tf.shape(encoding)[1]

    def highway_maxout_network(answer):
        span_encoding = start_and_end_encoding(encoding, answer)
        r_input = tf.concat([state, span_encoding], axis=1)
        r_input = tf.nn.dropout(r_input, keep_prob)
        r = tf.layers.dense(r_input, state_size, use_bias=False, activation=tf.tanh)
        r = tf.expand_dims(r, 1)
        r = tf.tile(r, (1, max_len, 1))
        highway_input = tf.concat([encoding, r], 2)
        logit = highway_maxout(highway_input, state_size, pool_size, keep_prob)
        logit = _maybe_mask_score(logit, document_length, -1e30)
        return logit

    with tf.variable_scope('start'):
        alpha = highway_maxout_network(answer)

    with tf.variable_scope('end'):
        updated_start = tf.argmax(alpha, axis=1, output_type=tf.int32)
        updated_answer = tf.stack([updated_start, answer[:, 1]], axis=1)
        beta = highway_maxout_network(updated_answer)

    return tf.stack([alpha, beta], axis=2)

