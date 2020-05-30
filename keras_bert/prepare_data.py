import codecs
import random

from bert.tokenization import FullTokenizer

MASK_TOKEN = "[MASK]"
CLASS_TOKEN = "[CLS]"
SENTENCE_SEPARATOR_TOKEN = "[SEP]"
NOT_TO_CHANGE = [CLASS_TOKEN, SENTENCE_SEPARATOR_TOKEN]


def create_tokens(text_file: str) -> [[str]]:
    data = codecs.open(text_file, 'r', 'utf-8')
    lines = data.readlines()

    count = 0
    texts = [[]]

    for line in lines:
        if not line.strip():
            count += 1
            texts.insert(count, [])
        else:
            texts[count].append(line.strip())

    tokens = []
    count = 0
    tokenizer = FullTokenizer(vocab_file='../data/bert_pl_model/vocab.txt')

    for text in texts:
        tokens.insert(count, [])
        tokens[count].append(CLASS_TOKEN)
        for sent in text:
            tokens[count] += (tokenizer.tokenize(sent))
            tokens[count].append(SENTENCE_SEPARATOR_TOKEN)

    return tokens


# Create list of tokens ids
def create_ids(tokens_list, max_len):
    tokenizer = FullTokenizer(vocab_file='../data/bert_pl_model/vocab.txt')
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
            if token == "[SEP]":
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
        for i in range(random.randint(3, 5)):
            index = random.randint(1, len(tokens) - 2)
            value = test[index]
            if value not in NOT_TO_CHANGE:
                test[index] = MASK_TOKEN
        test_inputs.append(test)
    return test_inputs
