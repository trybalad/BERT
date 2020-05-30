from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import numpy

from keras_bert.prepare_data import create_train_data, create_segments, create_ids, create_masks


def train_model(bert_model : Model, tokens, max_len : int, vocab_size : int):
    train_tokens = create_train_data(tokens)
    train_ids = numpy.array(create_ids(train_tokens, max_len))
    train_segments = numpy.array(create_segments(train_tokens, max_len))
    train_mask = numpy.array(create_masks(train_tokens, max_len))
    expected_ids = create_ids(tokens, max_len)

    bert_model.compile(optimizer=Adam(),
                       loss=categorical_crossentropy)

    expected_one_hot = numpy.array(one_hot(expected_ids,vocab_size))
    print(numpy.shape(train_ids))
    print(numpy.shape(expected_one_hot))

    bert_model.fit([train_ids, train_segments, train_mask], [expected_one_hot], batch_size=10, epochs=10)


def one_hot(expected_ids, vocab_size):
    result = []
    i = 0
    for row in expected_ids:
        token = []
        for token_id in row:
            expected_row = [0] * vocab_size
            expected_row[token_id] = 1
            token.append(expected_row)
        result.append(token)

    return result
