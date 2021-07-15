from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Model
from tensorflow.keras.backend import switch, zeros_like, floatx, sum, cast, epsilon, argmax, equal, not_equal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.optimizers import Adam

from keras_bert.tokenizer import Tokenizer


def train_model(bert_model: Model, max_len: int, tokenizer: Tokenizer, data_generator, val_generator=None,
                epochs=10, checkpoint_file_path=None, load_checkpoint=False, old_checkpoint=None, learning_rate=2e-5,
                learn_type="all"):
    print("Vocab size:", tokenizer.vocab_size)
    print("Max len of tokens:", max_len)

    training_model = prepare_pretrain_model_from_checkpoint(bert_model, tokenizer, checkpoint_file_path,
                                                            load_checkpoint,
                                                            old_checkpoint, learning_rate, learn_type)

    if checkpoint_file_path:
        checkpoint = [ModelCheckpoint(filepath=checkpoint_file_path, save_weights_only=True, verbose=1)]
    else:
        checkpoint = None

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=5, embeddings_freq=5)
    training_model.fit(data_generator, validation_data=val_generator, epochs=epochs,
                       callbacks=[tensorboard_callback, checkpoint])
    return training_model


def prepare_pretrain_model_from_checkpoint(bert_model: Model, tokenizer: Tokenizer, checkpoint_file_path=None,
                                           load_checkpoint=True, old_checkpoint=None, learning_rate=2e-5,
                                           learn_type="all"):
    training_model = prepare_training_model(bert_model, tokenizer, learn_type)
    loss, loss_weights, metrics = prepare_losses_and_metrics(learn_type)

    training_model.compile(optimizer=Adam(learning_rate), loss=loss, loss_weights=loss_weights, metrics=metrics)

    if load_checkpoint and old_checkpoint:
        training_model.load_weights(old_checkpoint)
    elif load_checkpoint and checkpoint_file_path:
        training_model.load_weights(checkpoint_file_path)

    print(training_model.summary())
    return training_model


def prepare_training_model(bert_model, tokenizer, training_type):
    outputs = []
    output_mlm = Dense(tokenizer.vocab_size, activation="softmax", name='mlm')(bert_model.output)
    output_nsp = Flatten()(bert_model.output)
    output_nsp = Dense(1, activation="sigmoid", name="nsp")(output_nsp)

    if training_type == "all":
        outputs.append(output_mlm)
        outputs.append(output_nsp)
    elif training_type == "mlm":
        outputs.append(output_mlm)
    elif training_type == "nsp":
        outputs.append(output_nsp)

    return Model(inputs=bert_model.inputs, outputs=outputs)


def prepare_losses_and_metrics(learn_type):
    if learn_type == "all":
        loss = {'mlm': masked_loss, 'nsp': binary_crossentropy}
        loss_weights = {'mlm': 0.6, 'nsp': 0.4}
        metrics = {'mlm': masked_accuracy, 'nsp': binary_accuracy}
        return loss, loss_weights, metrics
    elif learn_type == "mlm":
        loss = masked_loss
        metrics = masked_accuracy
        return loss, None, metrics
    elif learn_type == "nsp":
        loss = binary_crossentropy
        metrics = binary_accuracy
        return loss, None, metrics
    else:
        return None


def masked_loss(y_true, y_pred):
    max_args = argmax(y_true)
    mask = cast(not_equal(max_args, zeros_like(max_args)), dtype='float32')
    loss = switch(mask, categorical_crossentropy(y_true, y_pred, from_logits=True), zeros_like(mask, dtype=floatx()))
    return sum(loss) / (cast(sum(mask), dtype='float32') + epsilon())


def masked_accuracy(y_true, y_pred):
    max_args = argmax(y_true)
    mask = cast(not_equal(max_args, zeros_like(max_args)), dtype='float32')
    points = switch(mask, cast(equal(argmax(y_true, -1), argmax(y_pred, -1)), dtype='float32'),
                    zeros_like(mask, dtype=floatx()))
    return sum(points) / cast(sum(mask), dtype='float32')
