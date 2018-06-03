# Author: Harsh Kohli
# Date created: 5/22/2018

import numpy as np
from nltk import word_tokenize
import random


def get_word_indices(tokens, word_to_id_lookup):
    indices = []
    for token in tokens:
        lower_word = token.lower()
        if lower_word in word_to_id_lookup:
            indices.append(word_to_id_lookup[lower_word])
        else:
            indices.append(word_to_id_lookup['unk'])
    return indices


def get_closest_span_marco(para_tokens, answer_tokens):
    for index, para_word in enumerate(para_tokens):
        if (index + len(answer_tokens)) > len(para_tokens):
            break
        count = 0
        last_count = index
        for answer_word_no in range(len(answer_tokens)):
            if answer_tokens[answer_word_no].lower() == para_tokens[index + answer_word_no].lower():
                count = count + 1
                last_count = index + answer_word_no
        if float(count) / len(answer_tokens) > 0.65:
            return index, last_count
    return None, None


def get_batch_questions(dataset, iteration_no, config):
    start = (iteration_no * config['batch_size']) % (len(dataset))
    end = start + config['batch_size']
    if end < len(dataset):
        return [x for x in range(da)]
    else:
        batch = dataset[start: len(dataset)]
        batch.extend(dataset[0:(end - len(dataset))])


def get_adverserial_batch(dataset, iteration_no, word_to_id_lookup, config):
    start = (iteration_no * config['batch_size']) % (len(dataset))
    end = start + config['batch_size']
    if end < len(dataset):
        return get_adverserial_batch_data(dataset[start:end], word_to_id_lookup)
    else:
        batch = dataset[start: len(dataset)]
        batch.extend(dataset[0:(end - len(dataset))])
        return get_adverserial_batch_data(batch, word_to_id_lookup)


def get_adverserial_batch_data(batch, word_to_id_lookup):
    paragraphs, questions, answer_start, answer_end, para_sizes, questions_sizes = [], [], [], [], [], []
    for element in batch:
        for adverserial_element in element['AdversaryInfo']:
            paragraphs.append(adverserial_element['Indices'])
            questions.append(element['QuestionInfo']['QuestionIndices'])
            para_sizes.append(adverserial_element['Length'] + 3)
            questions_sizes.append(element['QuestionInfo']['QuestionLength'])
            answer_start.append(adverserial_element['Length'] + 2)
            answer_end.append(adverserial_element['Length'] + 2)
    return create_numpy_dict(pad_and_stack(paragraphs, max(para_sizes), word_to_id_lookup),
                             pad_and_stack(questions, max(questions_sizes), word_to_id_lookup), np.stack(answer_start),
                             np.stack(answer_end), np.stack(para_sizes), np.stack(questions_sizes))


def get_training_batch(dataset, best_adversaries, iteration_no, word_to_id_lookup, config):
    start = (iteration_no * config['batch_size']) % (len(dataset))
    end = start + config['batch_size']
    if end < len(dataset):
        return get_training_batch_data(dataset[start:end], best_adversaries, word_to_id_lookup)
    else:
        batch = dataset[start: len(dataset)]
        batch.extend(dataset[0:(end - len(dataset))])
        return get_training_batch_data(batch, best_adversaries, word_to_id_lookup)


def get_training_batch_data(batch, best_adversaries, word_to_id_lookup):
    paragraphs, questions, answer_start, answer_end, para_sizes, questions_sizes = [], [], [], [], [], []
    for element in batch:
        if element['Question'] in best_adversaries:
            adversary_indices = best_adversaries['Question']['Indices']
            para_indices = element['ParagraphInfo']['Indices']
            adversary_first = False
            if bool(random.getrandbits(1)):
                paragraphs.append(adversary_indices + para_indices)
                adversary_first = True
            else:
                paragraphs.append(para_indices + adversary_indices)
            para_sizes.append(best_adversaries['Question']['Length'] + element['ParagraphInfo']['Length'] + 3)
            answer = element['ParagraphInfo']['Answer']
            if answer_start is not None and answer_end is not None:
                if adversary_first:
                    answer_start.append(
                        best_adversaries['Question']['Length'] + element['ParagraphInfo']['AnswerStart'])
                    answer_end.append(
                        best_adversaries['Question']['Length'] + element['ParagraphInfo']['AnswerEnd'])
                else:
                    answer_start.append(element['ParagraphInfo']['AnswerStart'])
                    answer_end.append(element['ParagraphInfo']['AnswerStart'])
            elif answer.lower() == 'yes':
                answer_start.append(para_sizes[-1] - 3)
                answer_end.append(para_sizes[-1] - 3)
            elif answer.lower() == 'no':
                answer_start.append(para_sizes[-1] - 2)
                answer_end.append(para_sizes[-1] - 2)
            else:
                answer_start.append(para_sizes[-1] - 1)
                answer_end.append(para_sizes[-1] - 1)
        questions.append(element['QuestionInfo']['QuestionIndices'])
        questions_sizes.append(element['QuestionInfo']['QuestionLength'])
    return create_numpy_dict(pad_and_stack(paragraphs, max(para_sizes), word_to_id_lookup),
                             pad_and_stack(questions, max(questions_sizes), word_to_id_lookup), np.stack(answer_start),
                             np.stack(answer_end), np.stack(para_sizes), np.stack(questions_sizes))


def create_numpy_dict(paragraphs, questions, answer_starts, answer_ends, para_lengths, question_lengths):
    batch_info = {
        'paragraphs': paragraphs,
        'questions': questions,
        'answer_starts': np.array(answer_starts),
        'answer_ends': np.array(answer_ends),
        'para_lengths': np.array(para_lengths),
        'question_lengths': np.array(question_lengths)
    }
    return batch_info


def pad_and_stack(sequences, length, word_to_id_lookup):
    padded_sequence = []
    for sequence in sequences:
        padded_sequence.append(np.array(sequence + [word_to_id_lookup['***pad***']] * (length - len(sequence))))
    return np.stack(padded_sequence)


def cleanly_tokenize(text):
    return word_tokenize(text.replace("-", " - ").replace('–', ' – ').replace("''", '" ').replace("``", '" '))
