# Author: Harsh Kohli
# Date created: 5/15/2018

import yaml
import pickle
from utils.ioutils import read_marco_data, read_word_embeddings
from models.machine_comprehension import MCModel as Model

if __name__ == '__main__':
    config = yaml.safe_load(open('config.yml', 'r'))
    word_to_id_lookup, embeddings = read_word_embeddings(config['embedding_path'])
    config['vocab_size'] = embeddings.shape[0]
    config['embedding_size'] = embeddings.shape[1]

    serialized_data_file = open(config['preprocessed_data_path'], 'rb')
    data = pickle.load(serialized_data_file)
    train_data, dev_data = data['train_data'], data['dev_data']

    model = Model(config)
    model.train(train_data, dev_data, word_to_id_lookup, embeddings, config)
