# Author: Harsh Kohli
# Date created: 5/15/2018

from utils.ioutils import read_marco_data, read_word_embeddings
from models.machine_comprehension import MCModel as Model

if __name__ == '__main__':
    config = dict()

    config['task'] = 'marco'
    config['train_path'] = 'datasets/train_v2.1.json'
    config['dev_path'] = 'datasets/dev_v2.1.json'
    config['embedding_path'] = 'embeddings/glove.6B.100d.txt'

    word_to_id_lookup, embeddings = read_word_embeddings(config['embedding_path'])
    train_data = read_marco_data(config['train_path'], word_to_id_lookup)
    dev_data = read_marco_data(config['dev_path'], word_to_id_lookup)

    model = Model(config)
    model.train()
