# Author: Harsh Kohli
# Date created: 5/15/2018

import yaml
import pickle
import tensorflow as tf
from models.machine_comprehension import MCModel as Model

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml', 'r'))
    serialized_data_file = open(config['preprocessed_data_path'], 'rb')
    data = pickle.load(serialized_data_file)
    train_data, dev_data, word_to_id_lookup, embeddings = data['train_data'], data['dev_data'], data[
        'word_to_id_lookup'], data['embeddings']

    config['vocab_size'] = embeddings.shape[0]
    config['embedding_size'] = embeddings.shape[1]

    if config['task'] == 'squad':
        config['extra_vectors'] = 1
    elif config['task'] == 'marco':
        config['extra_vectors'] = 3
    else:
        raise ValueError('Invalid task type')

    sess = tf.Session()
    model = Model(config)
    model.initialize_word_embeddings(sess, embeddings)
    model.train(sess, train_data, dev_data, word_to_id_lookup, config)
