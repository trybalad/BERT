import codecs
import random

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils.data_utils import Sequence

from keras_bert.prepare_data import create_pretrain_data, create_segments, create_ids, create_masks, create_tokens, \
    create_tokens_with_nsr

"""
Data generator for training/validation of model

:param text_file: name of file from which sentences should be gathered
:param max_len: max len of input tokens list
:param vocab_size: size of used vocab
:param batch_size: size of each batch
"""


class DataGenerator(Sequence):
    def __init__(self, text_file: str, max_len, tokenizer, text_file_nsr=None, document_lines_size=None, batch_size=32,
                 create_nsr_output=True):
        self.max_len = max_len
        self.vocab_size = tokenizer.vocab_size
        self.text_file = text_file
        self.create_nsr_output = create_nsr_output
        if text_file_nsr is not None:
            file = codecs.open(text_file_nsr, 'r', 'utf-8')
            self.nsr_lines = file.readlines()
        else:
            self.nsr_lines = None
        self.tokenizer = tokenizer

        self.document_lines_size = document_lines_size
        self.start_index = 0
        self.end_index = batch_size
        self.batch_size = batch_size

    def __len__(self):
        if self.document_lines_size is not None:
            return int(np.ceil(self.document_lines_size / self.batch_size))
        else:
            count = 0
            file = codecs.open(self.text_file, 'r', 'utf-8')
            line = file.readline()
            while line:
                count += 1
                line = file.readline()
            self.document_lines_size = count
            return int(np.ceil(count / self.batch_size))

    def __getitem__(self, index):
        start = self.batch_size * index
        end = min(start + self.batch_size, self.document_lines_size)
        content_lines = self.get_content_lines(start, end)
        x, y = self.generate_data(content_lines)
        return x, y

    def get_content_lines(self, start, end):
        file = codecs.open(self.text_file, 'r', 'utf-8')
        content_lines = []

        for i in range(0, end):
            if i < start:
                file.readline()
            elif start <= i < end:
                content_lines.append(file.readline().strip())
            else:
                break
        return content_lines

    def generate_data(self, lines):
        if self.nsr_lines is None and not self.create_nsr_output:
            tokens = create_tokens(lines, self.tokenizer, self.max_len)
        elif self.nsr_lines is None and self.create_nsr_output:
            nsp_index = random.randint(0, int(np.ceil(self.document_lines_size / self.batch_size)) - 1)
            start = self.batch_size * nsp_index
            end = min(start + self.batch_size, self.document_lines_size)
            tokens, expected_nsr = create_tokens_with_nsr(lines, self.get_content_lines(start, end), self.tokenizer,
                                                          self.max_len)
        else:
            tokens, expected_nsr = create_tokens_with_nsr(lines, self.nsr_lines, self.tokenizer, self.max_len)
        train_tokens = create_pretrain_data(tokens, self.tokenizer)

        train_ids = np.array(create_ids(train_tokens, self.max_len, self.tokenizer))
        train_segments = np.array(create_segments(train_tokens, self.max_len))
        train_mask = np.array(create_masks(train_tokens, self.max_len))

        expected_ids = create_ids(tokens, self.max_len, self.tokenizer)
        expected_one_hot = [to_categorical(expected_id, self.vocab_size) for expected_id in expected_ids]

        # if self.nsr_lines is None and self.create_nsr_output:
        #     expected_nsr = create_nsr(train_segments)

        inputs = [np.array(train_ids), np.array(train_segments), np.array(train_mask)]

        if self.create_nsr_output:
            outputs = [np.array(expected_one_hot), np.array(expected_nsr)]
        else:
            outputs = [np.array(expected_one_hot)]

        return inputs, outputs
