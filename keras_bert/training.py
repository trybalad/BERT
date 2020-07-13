from tensorflow.keras import Model
from tensorflow.keras.backend import dot, transpose
from tensorflow.keras.layers import Lambda, TimeDistributed
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.optimizers import Adam

from keras_bert.data_generator import DataGenerator
from keras_bert.utils import plot_model_history


def train_model(bert_model: Model, max_len: int, vocab_size, batch_size, epochs):
    decoder = Lambda(lambda x: dot(x, transpose(bert_model.get_layer('Tokens_Embedding').weights[0])), name='lm_logits')
    output = TimeDistributed(decoder)(bert_model.outputs[0])

    training_model = Model(inputs=bert_model.inputs, outputs=[output])
    print(training_model.summary())
    print("Vocab size:", vocab_size)
    print("Max len of tokens:", max_len)
    training_model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=[categorical_accuracy])

    generator = DataGenerator("../data/Document.txt", max_len, vocab_size, batch_size=batch_size)
    val_generator = DataGenerator("../data/Validation.txt", max_len, vocab_size, batch_size=batch_size)

    history = training_model.fit_generator(generator=generator, validation_data=val_generator, epochs=epochs)
    plot_model_history(history)