from tensorflow.keras import Model
from tensorflow.keras.backend import switch, zeros_like, floatx, sum, cast, epsilon, argmax, equal, not_equal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from keras_bert.data_generator import DataGenerator
from keras_bert.tokenizer import Tokenizer


def train_model(bert_model: Model, max_len: int, tokenizer: Tokenizer, train_file, validate_file=None,
                training_data_length=None, validation_data_length=None, batch_size=10,
                epochs=10, checkpoint_file_path=None, load_checkpoint=False, old_checkpoint=None):
    output = Dense(tokenizer.vocab_size)(bert_model.outputs[0])
    training_model = Model(inputs=bert_model.inputs, outputs=[output])

    print(training_model.summary())
    print("Vocab size:", tokenizer.vocab_size)
    print("Max len of tokens:", max_len)

    training_model.compile(optimizer=Adam(), loss=tokens_loss,metrics=[tokens_accuracy])

    generator = DataGenerator(train_file, max_len, tokenizer.vocab_size,tokenizer, training_data_length,
                              batch_size=batch_size)

    if load_checkpoint and old_checkpoint:
        training_model.load_weights(old_checkpoint)
    elif load_checkpoint and checkpoint_file_path:
        training_model.load_weights(checkpoint_file_path)

    if validate_file:
        val_generator = DataGenerator(validate_file, max_len, tokenizer.vocab_size, tokenizer, validation_data_length,
                                      batch_size=batch_size)
    else:
        val_generator = None

    if checkpoint_file_path:
        checkpoint = [ModelCheckpoint(filepath=checkpoint_file_path, save_weights_only=True, verbose=1)]
    else:
        checkpoint = None

    history = training_model.fit(generator, validation_data=val_generator, epochs=epochs,
                                 callbacks=checkpoint)
    # plot_model_history(history)
    return training_model


def tokens_loss(y_true, y_pred):
    max_args = argmax(y_true)
    mask = cast(not_equal(max_args, zeros_like(max_args)), dtype='float32')
    loss = switch(mask, classification_loss(y_true, y_pred), zeros_like(mask, dtype=floatx()))
    return sum(loss) / (cast(sum(mask), dtype='float32') + epsilon())


def classification_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred, from_logits=True)


def tokens_accuracy(y_true, y_pred):
    max_args = argmax(y_true)
    mask = cast(not_equal(max_args, zeros_like(max_args)), dtype='float32')
    points = switch(mask, cast(equal(argmax(y_true, -1), argmax(y_pred, -1)), dtype='float32'),
                    zeros_like(mask, dtype=floatx()))
    return sum(points) / cast(sum(mask), dtype='float32')
