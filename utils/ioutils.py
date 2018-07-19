# Author: Harsh Kohli
# Date created: 5/15/2018

import json
import numpy as np
import random
import copy
from utils.data_processing import get_word_indices
from utils.data_processing import cleanly_tokenize, get_closest_span_marco, get_closest_span_squad

def read_data(config, word_to_id_lookup, type):
    if config['task'] == 'marco':
        if type == 'train':
            data = json.load(open(config['train_path'], 'r'))
            return read_marco_train_data(data, word_to_id_lookup)
        elif type == 'dev':
            data = json.load(open(config['dev_path'], 'r'))
            return read_marco_dev_data(data, word_to_id_lookup)
    elif config['task'] == 'squad':
        if type == 'train':
            data = json.load(open(config['train_path'], 'r'))
            return read_squad_train_data(data, word_to_id_lookup)
        elif type == 'dev':
            data = json.load(open(config['dev_path'], 'r'))
            return read_marco_dev_data(data, word_to_id_lookup)
    else:
        raise ValueError('Invalid task type')

def read_squad_train_data(data, word_to_id_lookup):
    dataset = []
    for datum in data['data']:
        for paragraph in datum['paragraphs']:
            para_tokens = cleanly_tokenize(paragraph['context'])
            para_indices = get_word_indices(para_tokens, word_to_id_lookup)
            all_data = []
            all_questions = []
            for sample in paragraph['qas']:
                if sample['is_impossible'] is True:
                    answer_start, answer_end = None, None
                    answer = 'no answer present.'
                elif sample['is_impossible'] is False:
                    answer = sample['answers'][0]['text']
                    answer_tokens = cleanly_tokenize(answer)
                    answer_start, answer_end = get_closest_span_squad(para_tokens, answer_tokens, sample['answers'][0]['answer_start'])
                    if answer_start is None or answer_end is None:
                        break
                else:
                    raise ValueError('Dont know whether to answer')
                all_data.append((
                    {'Tokens': para_tokens, 'Indices': para_indices, 'Length': (len(para_tokens)), 'Answer': answer,
                     'AnswerStart': answer_start, 'AnswerEnd': answer_end}))
                question = sample['question']
                question_tokens = cleanly_tokenize(question)
                question_indices = get_word_indices(question_tokens, word_to_id_lookup)
                all_questions.append({'QuestionTokens': question_tokens, 'QuestionIndices': question_indices,
                                 'QuestionLength': len(question_tokens), 'Question': question})
            for index, processed_datum in enumerate(all_data):
                adversaries = copy.deepcopy(all_data)
                adversaries.remove(processed_datum)
                for adversary in adversaries:
                    adversary['Answer'] = 'no answer present.'
                    adversary['AnswerStart'], adversary['AnswerEnd'] = None, None
                dataset.append({'ParagraphInfo': processed_datum, 'AdversaryInfo': adversaries, 'QuestionInfo': all_questions[index]})
    print('here')

def read_marco_train_data(data, word_to_id_lookup):
    dataset = []
    for index, passage_id in enumerate(data['passages']):
        answer = data['answers'][passage_id][0]
        answer_tokens = cleanly_tokenize(answer)
        good_cop = None
        bad_cop = []
        for passage_no, passage in enumerate(data['passages'][passage_id]):
            para_tokens = cleanly_tokenize(passage['passage_text'])
            para_indices = get_word_indices(para_tokens, word_to_id_lookup)
            if passage['is_selected'] == 1:
                if answer.lower() != 'yes' and answer.lower() != 'no' and answer.lower() != 'no answer present.':
                    answer_start, answer_end = get_closest_span_marco(para_tokens, answer_tokens)
                    if answer_start is None or answer_end is None:
                        continue
                    good_cop = (
                        {'Tokens': para_tokens, 'Indices': para_indices, 'Length': (len(para_tokens)), 'Answer': answer,
                         'AnswerStart': answer_start, 'AnswerEnd': answer_end})
                elif answer.lower() == 'yes' or answer.lower() == 'no':
                    answer_start, answer_end = None, None
                    good_cop = (
                        {'Tokens': para_tokens, 'Indices': para_indices, 'Length': (len(para_tokens)), 'Answer': answer,
                         'AnswerStart': answer_start, 'AnswerEnd': answer_end})
            else:
                bad_cop.append({'Tokens': para_tokens, 'Indices': para_indices, 'Length': len(para_tokens),
                                'Answer': 'no answer present.', 'AnswerStart': None, 'AnswerEnd': None})
        if good_cop is None or len(good_cop) == 0:
            transfer = random.choice(bad_cop)
            good_cop = transfer
            bad_cop.remove(transfer)
        question = data['query'][passage_id]
        question_tokens = cleanly_tokenize(question)
        question_indices = get_word_indices(question_tokens, word_to_id_lookup)
        question_info = {'QuestionTokens': question_tokens, 'QuestionIndices': question_indices,
                         'QuestionLength': len(question_tokens), 'Question': question}
        dataset.append({'ParagraphInfo': good_cop, 'AdversaryInfo': bad_cop, 'QuestionInfo': question_info})
    return dataset


def read_marco_dev_data(file_path, word_to_id_lookup):
    data = json.load(open(file_path, 'r'))
    dataset = []
    for index, passage_id in enumerate(data['passages']):
        answer = data['answers'][passage_id][0]
        answer_tokens = cleanly_tokenize(answer)
        question = data['query'][passage_id]
        question_tokens = cleanly_tokenize(question)
        question_indices = get_word_indices(question_tokens, word_to_id_lookup)
        question_info = {'QuestionTokens': question_tokens, 'QuestionIndices': question_indices,
                         'QuestionLength': len(question_tokens)}
        for passage_no, passage in enumerate(data['passages'][passage_id]):
            para_tokens = cleanly_tokenize(passage['passage_text'])
            para_indices = get_word_indices(para_tokens, word_to_id_lookup)
            if passage['is_selected'] == 1:
                if answer.lower() != 'yes' and answer.lower() != 'no' and answer.lower() != 'no answer present.':
                    answer_start, answer_end = get_closest_span_marco(para_tokens, answer_tokens)
                    if answer_start is None or answer_end is None:
                        continue
                elif answer.lower() == 'yes' or answer.lower() == 'no':
                    answer_start, answer_end = None, None
                query_data = {'Tokens': para_tokens, 'Indices': para_indices, 'Length': (len(para_tokens)),
                              'Answer': answer,
                              'AnswerStart': answer_start, 'AnswerEnd': answer_end}
            else:
                query_data = {'Tokens': para_tokens, 'Indices': para_indices, 'Length': len(para_tokens),
                              'Answer': 'no answer present.', 'AnswerStart': None, 'AnswerEnd': None}
            dataset.append({'ParagraphInfo': query_data, 'QuestionInfo': question_info})
    return dataset


def read_word_embeddings(file_path):
    f = open(file_path, "r", encoding='utf8')
    word_to_id_lookup = {}
    vectors = []
    word_to_id_lookup['***unk***'] = 0
    word_to_id_lookup['***pad***'] = 1
    word_to_id_lookup['***true***'] = 2
    word_to_id_lookup['***false***'] = 3
    word_to_id_lookup['***no_answer***'] = 4
    for index, line in enumerate(f.readlines()):
        info = line.strip().split(' ')
        word = info[0]
        vector = np.array(info[1:])
        word_to_id_lookup[word] = index + 5
        vectors.append(vector)
    unk_vector = np.random.uniform(-0.1, 0.1, vector.size)
    pad_vector = np.random.uniform(-0.1, 0.1, vector.size)
    true_vector = np.random.uniform(-0.1, 0.1, vector.size)
    false_vector = np.random.uniform(-0.1, 0.1, vector.size)
    no_answer = np.random.uniform(-0.1, 0.1, vector.size)
    vectors = [unk_vector] + [pad_vector] + [true_vector] + [false_vector] + [no_answer] + vectors
    return word_to_id_lookup, np.array(vectors)
