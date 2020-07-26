from tensorflow.keras import Model
from tensorflow.keras.backend import dot, transpose
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Lambda, TimeDistributed
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from keras_bert.data_generator import DataGenerator
from keras_bert.tokenizer import Tokenizer
from keras_bert.utils import plot_model_history


def train_model(bert_model: Model, max_len: int, tokenizer: Tokenizer, train_file, validate_file=None,
                training_data_length=1000, validation_data_length=1000, batch_size=10,
                epochs=10, checkpoint_file_path=None, load_checkpoint=False, old_checkpoint=None):
    decoder = Lambda(lambda x: dot(x, transpose(bert_model.get_layer('Tokens_Embedding').weights[0])), name='lm_logits')
    output = TimeDistributed(decoder)(bert_model.outputs[0])

    training_model = Model(inputs=bert_model.inputs, outputs=[output])

    print(training_model.summary())
    print("Vocab size:", tokenizer.vocab_size)
    print("Max len of tokens:", max_len)

    training_model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])

    generator = DataGenerator(train_file, max_len, tokenizer.vocab_size, training_data_length, tokenizer,
                              batch_size=batch_size)

    if load_checkpoint and old_checkpoint:
        training_model.load_weights(old_checkpoint)
    elif load_checkpoint and checkpoint_file_path:
        training_model.load_weights(checkpoint_file_path)

    if validate_file:
        val_generator = DataGenerator(validate_file, max_len, tokenizer.vocab_size, validation_data_length, tokenizer,
                                      batch_size=batch_size)
    else:
        val_generator = None

    if checkpoint_file_path:
        checkpoint = [ModelCheckpoint(filepath=checkpoint_file_path, save_weights_only=True, verbose=1)]
    else:
        checkpoint = None

    history = training_model.fit(generator, validation_data=val_generator, epochs=epochs,
                                           callbacks=checkpoint, batch_size=batch_size)
    plot_model_history(history)
