import codecs

import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.utils.data_utils import Sequence

from keras_bert.prepare_data import create_segments, create_ids, create_masks, create_tokens


class TuningDataGenerator(Sequence):
    def __init__(self, text_file, max_len, tokenizer, document_lines_size=None, batch_size=32, tuning_type='ar'):
        self.max_len = max_len
        self.vocab_size = tokenizer.vocab_size
        self.text_file = text_file
        self.tokenizer = tokenizer
        self.tuning_type = tuning_type

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

        text, outputs = self.parse_lines(lines)
        inputs = self.create_inputs(text)

        return inputs, outputs

    def parse_lines(self, lines):
        texts = []
        outputs = []

        if self.tuning_type == 'ar':
            for line in lines:
                splitted = line.split('\t')
                if splitted.__len__() == 2:
                    texts.append(splitted[0])
                    ar_output = float(splitted[1])
                    output = np.array(to_categorical(ar_output - 1, 5))
                    outputs.append(output)
                # if splitted size is equal to 1 its just number without text so we skip it
        elif self.tuning_type == 'dyk':
            for line in lines:
                splitted = line.split('\t')
                if splitted.__len__() == 4:
                    texts.append(splitted[1] + splitted[2])
                    dyk_output = int(splitted[3])
                    outputs.append(dyk_output)
        elif self.tuning_type == 'cbd':
            for line in lines:
                splitted = line.split('\t')
                if splitted.__len__() == 2:
                    texts.append(splitted[0])
                    cbd_output = int(splitted[1])
                    outputs.append(cbd_output)
        elif self.tuning_type == 'cdsc':
            for line in lines:
                splitted = line.split('\t')
                if splitted.__len__() == 4:
                    texts.append(splitted[1] + splitted[2])
                    cdsc_output = splitted[3]

                    output = 5
                    if cdsc_output == "NEUTRAL":
                        output = 0
                    elif cdsc_output == "CONTRADICTION":
                        output = 1
                    elif cdsc_output == "ENTAILMENT":
                        output = 2
                    outputs.append(np.array(to_categorical(output, 3)))

        return texts, np.array(outputs)

    def create_inputs(self, texts):
        tokens = create_tokens(texts, self.tokenizer, self.max_len)
        ids = np.array(create_ids(tokens, self.max_len, self.tokenizer))
        segments = np.array(create_segments(tokens, self.max_len))
        mask = np.array(create_masks(tokens, self.max_len))

        return [np.array(ids), np.array(segments), np.array(mask)]
