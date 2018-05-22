# Author: Harsh Kohli
# Date created: 5/22/2018

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
    for index,para_word in enumerate(para_tokens):
        if (index + len(answer_tokens)) > len(para_tokens):
            break
        count = 0
        last_count = index
        for answer_word_no in range(len(answer_tokens)):
            if answer_tokens[answer_word_no].lower() == para_tokens[index + answer_word_no].lower():
                count = count + 1
                last_count = index + answer_word_no
        if float(count)/len(answer_tokens) > 0.65:
            return index, last_count
    return None, None

def cleanly_tokenize(text):
    return word_tokenize(text.replace("-", " - ").replace('–', ' – ').replace("''", '" ').replace("``", '" '))
