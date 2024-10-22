# Author: Harsh Kohli
# Date created: 5/22/2018

import numpy as np
from nltk import word_tokenize
import random
import copy


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


def get_closest_span_squad(para_tokens, answer_tokens, answer_start):
    character_count = 0
    for index, para_word in enumerate(para_tokens):
        character_count = character_count + len(para_word) + 1
        if character_count > answer_start - 50:
            broken = False
            for answer_word_no, answer_word in enumerate(answer_tokens):
                if answer_word != para_tokens[index + answer_word_no]:
                    broken = True
                    break
            if broken:
                continue
            return index, index + len(answer_tokens)
    return None, None


def populate_id_to_answer(answers, batch_info, id_to_answer_map, batch):
    for index, answer in enumerate(answers):
        if answer[0] == batch_info['para_lengths'][index]:
            id_to_answer_map[batch[index]['ParagraphInfo']['ID']] = ''
        else:
            one_answer = ''
            for word_index in range(answer[0], answer[1] + 1):
                if word_index == len(batch[index]['ParagraphInfo']['Tokens']):
                    break
                one_answer = one_answer + batch[index]['ParagraphInfo']['Tokens'][word_index] + ' '
            id_to_answer_map[batch[index]['ParagraphInfo']['ID']] = one_answer.strip()


def process_squad_para(paragraph, word_to_id_lookup, mode):
    para_tokens = cleanly_tokenize(paragraph['context'])
    para_indices = get_word_indices(para_tokens, word_to_id_lookup)
    all_data = []
    all_questions = []
    for sample in paragraph['qas']:
        if sample['id'] == '56de1563cffd8e1900b4b5c4':
            print('found')
        if sample['is_impossible'] is True:
            answer_start, answer_end = None, None
            answer = 'no answer present.'
        elif sample['is_impossible'] is False:
            answer = sample['answers'][0]['text']
            answer_tokens = cleanly_tokenize(answer)
            answer_start, answer_end = get_closest_span_squad(para_tokens, answer_tokens,
                                                              sample['answers'][0]['answer_start'])
            if answer_start is None or answer_end is None:
                if mode == 'train':
                    continue
        else:
            raise ValueError('Dont know whether to answer')
        all_data.append((
            {'Tokens': para_tokens, 'Indices': para_indices, 'Length': (len(para_tokens)), 'Answer': answer,
             'AnswerStart': answer_start, 'AnswerEnd': answer_end, 'ID': sample['id']}))
        question = sample['question']
        question_tokens = cleanly_tokenize(question)
        question_indices = get_word_indices(question_tokens, word_to_id_lookup)
        all_questions.append({'QuestionTokens': question_tokens, 'QuestionIndices': question_indices,
                              'QuestionLength': len(question_tokens), 'Question': question})
    adversarial_mod = ((
        {'Tokens': para_tokens, 'Indices': para_indices, 'Length': (len(para_tokens)), 'Answer': 'no answer present.',
         'AnswerStart': None, 'AnswerEnd': None}))
    return all_data, all_questions, adversarial_mod


def get_batch_input(dataset, iteration_no, batch_size, wraparound):
    start = (iteration_no * batch_size) % (len(dataset))
    end = start + batch_size
    if end < len(dataset):
        batch = dataset[start:end]
    else:
        batch = dataset[start: len(dataset)]
        if wraparound is True:
            batch.extend(dataset[0:(end - len(dataset))])
    return batch


def get_adversarial_batch(batch, word_to_id_lookup, extra_vectors):
    paragraphs, questions, answer_start, answer_end, para_sizes, questions_sizes, ids = [], [], [], [], [], [], []
    for element in batch:
        for adversarial_element in element['AdversaryInfo']:
            update_default_fields(adversarial_element, paragraphs, questions, answer_start, answer_end, para_sizes,
                                  questions_sizes, adversarial_element['Length'] + extra_vectors - 1,
                                  adversarial_element['Length'] + extra_vectors - 1, element['QuestionInfo'],
                                  extra_vectors, ids)
    return create_numpy_dict(
        pad_and_stack(add_extra_embeddings(paragraphs, word_to_id_lookup, extra_vectors), max(para_sizes),
                      word_to_id_lookup),
        pad_and_stack(questions, max(questions_sizes), word_to_id_lookup), np.stack(answer_start),
        np.stack(answer_end), np.stack(para_sizes), np.stack(questions_sizes))


def add_extra_embeddings(paragraphs, word_to_id_lookup, extra_vectors):
    modified_paragraphs = []
    for old_paragraph in paragraphs:
        paragraph = copy.deepcopy(old_paragraph)
        if extra_vectors == 3:
            paragraph.append(word_to_id_lookup['***true***'])
            paragraph.append(word_to_id_lookup['***false***'])
            paragraph.append(word_to_id_lookup['***no_answer***'])
        elif extra_vectors == 1:
            paragraph.append(word_to_id_lookup['***no_answer***'])
        else:
            raise ValueError('Unknown extra configuration')
        modified_paragraphs.append(paragraph)
    return modified_paragraphs


def get_dev_batch(batch, word_to_id_lookup, extra_vectors):
    paragraphs, questions, answer_start, answer_end, para_sizes, questions_sizes, ids = [], [], [], [], [], [], []
    for element in batch:
        start = element['ParagraphInfo']['AnswerStart']
        end = element['ParagraphInfo']['AnswerEnd']
        if start is None or end is None:
            start, end = yes_no_dont_know(element['ParagraphInfo']['Answer'],
                                          element['ParagraphInfo']['Length'] + extra_vectors, extra_vectors)
        update_default_fields(element['ParagraphInfo'], paragraphs, questions, answer_start, answer_end, para_sizes,
                              questions_sizes,
                              start, end, element['QuestionInfo'], extra_vectors, ids)
    return create_numpy_dict(
        pad_and_stack(add_extra_embeddings(paragraphs, word_to_id_lookup, extra_vectors), max(para_sizes),
                      word_to_id_lookup),
        pad_and_stack(questions, max(questions_sizes), word_to_id_lookup), np.stack(answer_start),
        np.stack(answer_end), np.stack(para_sizes), np.stack(questions_sizes))


def update_default_fields(element, paragraphs, questions, answer_start, answer_end, para_sizes, questions_sizes, start,
                          end, question_info, extra_vectors, ids):
    paragraphs.append(element['Indices'])
    para_sizes.append(element['Length'] + extra_vectors)
    answer_start.append(start)
    answer_end.append(end)
    questions.append(question_info['QuestionIndices'])
    questions_sizes.append(question_info['QuestionLength'])
    if 'ID' in element:
        ids.append(element['ID'])


def get_training_batch(batch, best_adversaries, word_to_id_lookup, extra_vectors):
    paragraphs, questions, answer_starts, answer_ends, para_sizes, questions_sizes = [], [], [], [], [], []
    for element in batch:
        if element['QuestionInfo']['Question'] in best_adversaries:
            best_adversary = best_adversaries[element['QuestionInfo']['Question']]
            adversary_indices = best_adversary['Indices']
            para_indices = element['ParagraphInfo']['Indices']
            adversary_first = False
            if bool(random.getrandbits(1)):
                paragraphs.append(adversary_indices + para_indices)
                adversary_first = True
            else:
                paragraphs.append(para_indices + adversary_indices)
            para_sizes.append(best_adversary['Length'] + element['ParagraphInfo']['Length'] + extra_vectors)
            answer = element['ParagraphInfo']['Answer']
            answer_start = element['ParagraphInfo']['AnswerStart']
            answer_end = element['ParagraphInfo']['AnswerEnd']
            if answer_start is not None and answer_end is not None:
                if adversary_first is True:
                    answer_starts.append(best_adversary['Length'] + answer_start)
                    answer_ends.append(best_adversary['Length'] + answer_end)
                else:
                    answer_starts.append(answer_start)
                    answer_ends.append(answer_end)
            else:
                start, end = yes_no_dont_know(answer, para_sizes[-1], extra_vectors)
                answer_starts.append(start)
                answer_ends.append(end)
        questions.append(element['QuestionInfo']['QuestionIndices'])
        questions_sizes.append(element['QuestionInfo']['QuestionLength'])
    return create_numpy_dict(
        pad_and_stack(add_extra_embeddings(paragraphs, word_to_id_lookup, extra_vectors), max(para_sizes),
                      word_to_id_lookup),
        pad_and_stack(questions, max(questions_sizes), word_to_id_lookup), np.stack(answer_starts),
        np.stack(answer_ends), np.stack(para_sizes), np.stack(questions_sizes))


def yes_no_dont_know(answer, para_size, extra_vectors):
    if extra_vectors == 3:
        if answer.lower() == 'yes':
            start = (para_size - 3)
            end = (para_size - 3)
        elif answer.lower() == 'no':
            start = (para_size - 2)
            end = (para_size - 2)
        else:
            start = (para_size - 1)
            end = (para_size - 1)
    else:
        start = (para_size - 1)
        end = (para_size - 1)
    return start, end


def get_strongest_adversaries(batch, predicted_starts, predicted_ends):
    index = 0
    start_end_sums = np.array(predicted_starts) + np.array(predicted_ends)
    strongest_adversaries = {}
    for element in batch:
        min_prob = float('inf')
        best_adversary = None
        for adversary in element['AdversaryInfo']:
            probability_of_no_answer = start_end_sums[index][adversary['Length']]
            if probability_of_no_answer < min_prob:
                min_prob = probability_of_no_answer
                best_adversary = adversary
            index = index + 1
        if best_adversary is not None:
            strongest_adversaries[element['QuestionInfo']['Question']] = best_adversary
    return strongest_adversaries


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
