import random
from math import floor, ceil

import numpy as np

"""
Class preparing data for model.
Provided methods for creating:
Tokens - List of ids of tokens created from words of input sentence.
Segments - List of ids, 0 - for padding, 1 - for first sentence, 2 - 2nd sentence etc.
Masks - Mask of used indexes 0 - padding, 1 - used index
Training data - Creates [MASK] for 15% of inputs ids for MLM(Masked language modeling) task
"""

MASK_TOKEN = "[MASK]"
CLASS_TOKEN = "[CLS]"
SENTENCE_SEPARATOR_TOKEN = "[SEP]"
NOT_TO_CHANGE = [CLASS_TOKEN, SENTENCE_SEPARATOR_TOKEN]


# Creates tokens from lines of text
def create_tokens(lines, tokenizer, max_len) -> [[str]]:
    tokens = []
    count = 0

    for text in lines:
        if not text:
            continue

        sentence_count = 0
        tokens.insert(count, [])
        tokens[count].append(CLASS_TOKEN)

        for sent in tokenizer.split_sentences(text):
            tokens[count] += (tokenizer.convert_to_tokens(sent.text))
            tokens[count].append(SENTENCE_SEPARATOR_TOKEN)
            sentence_count += 1
            if sentence_count == 2:
                break

        if len(tokens[count]) > max_len:
            tokens[count] = tokens[count][0:max_len]

    return tokens


def count_data_len(line, tokenizer):
    count = 0
    for sent in tokenizer.split_sentences(line):
        count += 1
    return ceil(count / 2)


# Creates tokens from lines of text
def create_tokens_with_nsr(lines, nsr_lines, tokenizer, max_len) -> [[str]]:
    tokens = []
    nsr_pred = []
    count = 0
    index_nsr = 0

    for text in lines:
        if not text:
            continue

        should_generate = random.randint(0, 1)
        sentence_count = 0
        tokens.insert(count, [])
        tokens[count].append(CLASS_TOKEN)

        for sent in tokenizer.split_sentences(text):
            tokens[count] += (tokenizer.convert_to_tokens(sent.text))
            tokens[count].append(SENTENCE_SEPARATOR_TOKEN)
            sentence_count += 1
            if sentence_count == 2:
                nsr_pred.append(1)
                break
            elif should_generate == 1 and sentence_count == 1:
                index_nsr = random.randint(0, len(nsr_lines) - 1)
                sents = tokenizer.split_sentences(nsr_lines[index_nsr])
                tokens[count] += (tokenizer.convert_to_tokens(next(sents).text))
                tokens[count].append(SENTENCE_SEPARATOR_TOKEN)
                sentence_count += 1
                nsr_pred.append(0)

                break
        if sentence_count == 1:
            nsr_pred.append(0)

        if len(tokens[count]) > max_len:
            tokens[count] = tokens[count][0:max_len]

    return tokens, nsr_pred


# Create list of tokens ids
def create_ids(tokens_list, max_len, tokenizer):
    ids_list = []
    for tokens in tokens_list:
        ids_list.append(tokenizer.convert_tokens_to_ids(tokens) + [0] * (max_len - len(tokens)))
    return ids_list


# Create segment ids list for next sentence prediction
def create_segments(tokens_list, max_len):
    segments_list = []
    for tokens in tokens_list:
        count = 1
        segments = []
        for token in tokens:
            segments.append(count)
            if token == SENTENCE_SEPARATOR_TOKEN:
                count += 1

        # Add padding
        segments = segments + [0] * (max_len - len(tokens))
        segments_list.append(segments)

    return segments_list


# create_nsr based on segments
def create_nsr(segments_list):
    nsr_list = []
    for segments in segments_list:
        nsr_list.append(np.amax(segments) - 1)

    return nsr_list


# Creates masks 1 - significant data, 0 - padding
def create_masks(tokens_list, max_len):
    masks_list = []
    for tokens in tokens_list:
        masks_list.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
    return masks_list


# Creates test data
def create_pretrain_data(tokens_list, tokenizer):
    test_inputs = []
    for tokens in tokens_list:
        test_data = tokens.copy()
        mode = random.randint(1, 10)
        if mode <= 8:
            mask_tokens(test_data)
        elif mode == 9:
            create_random_tokens(test_data, tokenizer)

        test_inputs.append(test_data)
    return test_inputs


def mask_tokens(test_data):
    for i in range(floor(len(test_data) * 0.15)):
        index = random.randint(1, len(test_data) - 2)
        value = test_data[index]
        if value not in NOT_TO_CHANGE:
            test_data[index] = MASK_TOKEN


def create_random_tokens(test_data, tokenizer):
    for i in range(floor(len(test_data) * 0.1)):
        index = random.randint(1, len(test_data) - 2)
        value = test_data[index]
        if value not in NOT_TO_CHANGE:
            test_data[index] = tokenizer.index2word[random.randint(4, tokenizer.vocab_size - 1)]


# Reverts one coded ids of tokens to tokens form
def translate_ids(ids, tokenizer):
    tokens = []
    for i in range(len(ids[0][0])):
        decoded_datum = np.argmax(ids[0][0][i])
        tokens.append(tokenizer.convert_to_token(decoded_datum))
    return tokens
