import codecs

import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import to_categorical
from keras_bert.prepare_data import create_train_data, create_segments, create_ids, create_masks, create_tokens

"""
Data generator for training/validation of model

:param text_file: name of file from which sentences should be gathered
:param max_len: max len of input tokens list
:param vocab_size: size of used vocab
:param batch_size: size of each batch
"""


class DataGenerator(Sequence):
    def __init__(self, text_file: str, max_len, vocab_size, tokenizer, batch_size=32):
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.text_file = codecs.open(text_file, 'r', 'utf-8')
        self.lines = self.text_file.readlines()
        self.remove_empty_lines()
        self.tokenizer = tokenizer

        self.document_lines_count = len(self.lines)
        self.start_index = 0
        self.end_index = batch_size
        self.batch_size = batch_size

    def __len__(self):
        return int(np.floor(self.document_lines_count / self.batch_size))

    def __getitem__(self, index):
        start = self.batch_size * index
        end = start + self.batch_size
        content_lines = self.lines[start:end]
        x, y = self.generate_data(content_lines)
        return x, y

    def remove_empty_lines(self):
        self.lines = [[line.strip()] for line in self.lines if line.strip() != '']

    def generate_data(self, lines):
        tokens = create_tokens(lines, self.tokenizer)
        train_tokens = create_train_data(tokens)

        train_ids = np.array(create_ids(train_tokens, self.max_len, self.tokenizer))
        train_segments = np.array(create_segments(train_tokens, self.max_len))
        train_mask = np.array(create_masks(train_tokens, self.max_len))

        expected_ids = create_ids(tokens, self.max_len, self.tokenizer)
        expected_one_hot = [to_categorical(expected_id, self.vocab_size) for expected_id in expected_ids]

        return [train_ids, train_segments, train_mask], [expected_one_hot]