# Author: Harsh Kohli
# Date created: 5/22/2018

import numpy as np
from nltk import word_tokenize


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


def get_batch(dataset, iteration_no, word_to_id_lookup, config):
    start = (iteration_no * config['batch_size']) % (len(dataset))
    end = start + config['batch_size']
    if end < len(dataset):
        return get_relevant_batch_data(dataset[start:end], word_to_id_lookup)
    else:
        batch = dataset[start: len(dataset)]
        batch.extend(dataset[0:(end - len(dataset))])
        return get_relevant_batch_data(batch, word_to_id_lookup)


def get_relevant_batch_data(batch, word_to_id_lookup):
    paragraphs, questions, para_labels, answer_start, answer_end, para_sizes, questions_sizes = [], [], [], [], [], [], []
    max_para_len, max_question_len = 0, 0
    for element in batch:
        sizes, indices = [], []
        for paragraph in element['Paragraphs']:
            length = paragraph['Length']
            sizes.append(length)
            if length > max_para_len:
                max_para_len = length
            indices.append(paragraph['Indices'])
        paragraphs.append(indices)
        para_sizes.append(np.array(sizes))
        questions.append(element['Question']['QuestionIndices'])
        question_length = element['Question']['QuestionLength']
        questions_sizes.append(question_length)
        if question_length > max_question_len:
            max_question_len = question_length
        answer_start.append(element['AnswerStart'])
        answer_end.append(element['AnswerEnd'])
        para_labels.append(element['AnswerPassage'])
    padded_paragraphs = []
    for paragraph in paragraphs:
        padded_paragraphs.append(pad_and_stack(paragraph, max_para_len, word_to_id_lookup))
    return create_numpy_dict(np.array(padded_paragraphs), pad_and_stack(questions, max_question_len, word_to_id_lookup),
                             para_labels, answer_start, answer_end,
                             para_sizes, questions_sizes)


def create_numpy_dict(paragraphs, questions, para_labels, answer_starts, answer_ends, para_lengths, question_lengths):
    batch_info = {
        'paragraphs': paragraphs,
        'questions': questions,
        'para_labels': np.array(para_labels),
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
