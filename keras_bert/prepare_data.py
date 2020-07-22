import random
from math import floor

from bert.tokenization import FullTokenizer

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


def create_tokens(lines, tokenizer) -> [[str]]:
    tokens = []
    count = 0

    for text in lines:
        tokens.insert(count, [])
        tokens[count].append(CLASS_TOKEN)
        for sent in text:
            tokens[count] += (tokenizer.tokenize(sent))
            tokens[count].append(SENTENCE_SEPARATOR_TOKEN)

    return tokens


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
        count = 0
        segments = []
        for token in tokens:
            segments.append(count)
            if token == SENTENCE_SEPARATOR_TOKEN:
                count += 1

        # Add padding
        segments = segments + [0] * (max_len - len(tokens))
        segments_list.append(segments)

    return segments_list


# Creates masks 1 - significant data, 0 - padding
def create_masks(tokens_list, max_len):
    masks_list = []
    for tokens in tokens_list:
        masks_list.append([1] * len(tokens) + [0] * (max_len - len(tokens)))
    return masks_list


# Creates test data
def create_train_data(tokens_list):
    test_inputs = []
    for tokens in tokens_list:
        test = tokens.copy()
        for i in range(floor(len(test) * 0.15)):
            index = random.randint(1, len(tokens) - 2)
            value = test[index]
            if value not in NOT_TO_CHANGE:
                test[index] = MASK_TOKEN
        test_inputs.append(test)
    return test_inputs
