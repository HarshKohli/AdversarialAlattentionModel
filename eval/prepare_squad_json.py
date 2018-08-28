# Author: Harsh Kohli
# Date created: 7/30/2018

import yaml
import json
import tensorflow as tf
from utils.ioutils import read_word_embeddings, read_data
from models.machine_comprehension import MCModel as Model

if __name__ == '__main__':

    config = yaml.safe_load(open('config.yml', 'r'))
    word_to_id_lookup, embeddings = read_word_embeddings(config['embedding_path'])
    test_data = read_data(config, word_to_id_lookup, 'test')
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
    id_to_answer_map = model.test(sess, test_data, word_to_id_lookup, config)
    with open(config['test_output'], 'w') as outfile:
        json.dump(id_to_answer_map, outfile)
