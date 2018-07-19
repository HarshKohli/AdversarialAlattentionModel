# Author: Harsh Kohli
# Date created: 5/24/2018

import yaml
from utils.ioutils import read_word_embeddings, read_data
import pickle

config = yaml.safe_load(open('config.yml', 'r'))
word_to_id_lookup, embeddings = read_word_embeddings(config['embedding_path'])

train_data = read_data(config, word_to_id_lookup, 'train')
dev_data = read_data(config, word_to_id_lookup, 'dev')

all_data = {'train_data': train_data, 'dev_data': dev_data, 'word_to_id_lookup': word_to_id_lookup,
            'embeddings': embeddings}
pickle_file = open(config['preprocessed_data_path'], 'wb')
pickle.dump(all_data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
pickle_file.close()
