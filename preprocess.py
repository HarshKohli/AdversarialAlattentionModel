# Author: Harsh Kohli
# Date created: 5/24/2018

import yaml
from utils.ioutils import read_word_embeddings, read_marco_train_data, read_marco_dev_data
import pickle

config = yaml.safe_load(open('config.yml', 'r'))
word_to_id_lookup, embeddings = read_word_embeddings(config['embedding_path'])

primary_train_data, adverserial_train_data = read_marco_train_data(config['train_path'], word_to_id_lookup)
dev_data = read_marco_dev_data(config['dev_path'], word_to_id_lookup)

all_data = {'primary_train_data': primary_train_data, 'adverserial_train_data': adverserial_train_data,
            'dev_data': dev_data, 'word_to_id_lookup': word_to_id_lookup,
            'embeddings': embeddings}
pickle_file = open(config['preprocessed_data_path'], 'wb')
pickle.dump(all_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
pickle_file.close()
