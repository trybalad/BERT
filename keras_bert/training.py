from tensorflow.keras import Model
from tensorflow.keras.backend import switch, zeros_like, floatx, sum, cast, epsilon, argmax, equal, not_equal
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import binary_accuracy, categorical_accuracy
from tensorflow.keras.optimizers import Adam

from keras_bert.tokenizer import Tokenizer


def prepare_losses_and_metrics(learn_type):
    if learn_type == "all":
        loss = {'mlp': mlp_loss, 'nsr': binary_crossentropy}
        loss_weights = {'mlp': 0.9, 'nsr': 0.1}
        metrics = {'mlp': [mlp_accuracy, categorical_accuracy], 'nsr': binary_accuracy}
        return loss, loss_weights, metrics
    elif learn_type == "mlp":
        loss = categorical_crossentropy #mlp_loss
        metrics = [mlp_accuracy, categorical_accuracy]
        return loss, None, metrics
    elif learn_type == "nsr":
        loss = binary_crossentropy
        metrics = binary_accuracy
        return loss, None, metrics
    else:
        return None


def train_model(bert_model: Model, max_len: int, tokenizer: Tokenizer, data_generator, val_generator=None,
                epochs=10, checkpoint_file_path=None, load_checkpoint=False, old_checkpoint=None, learning_rate=0.05,
                learn_type="all"):
    training_model = prepare_training_model(bert_model, tokenizer, learn_type)

    print("Vocab size:", tokenizer.vocab_size)
    print("Max len of tokens:", max_len)

    loss, loss_weights, metrics = prepare_losses_and_metrics(learn_type)

    print(training_model.summary())
    training_model.compile(optimizer=Adam(learning_rate), loss=loss, loss_weights=loss_weights, metrics=metrics)

    if load_checkpoint and old_checkpoint:
        training_model.load_weights(old_checkpoint)
    elif load_checkpoint and checkpoint_file_path:
        training_model.load_weights(checkpoint_file_path)

    if checkpoint_file_path:
        checkpoint = [ModelCheckpoint(filepath=checkpoint_file_path, save_weights_only=True, verbose=1)]
    else:
        checkpoint = None

    history = training_model.fit_generator(data_generator, validation_data=val_generator, epochs=epochs,
                                           callbacks=checkpoint)
    # plot_model_history(history)
    return training_model


def prepare_training_model(bert_model, tokenizer, training_type):
    outputs = []
    output_mlp = Dense(tokenizer.vocab_size, activation="softmax", name='mlp')(bert_model.output)
    output_nsr = Flatten()(bert_model.output)
    output_nsr = Dense(1, activation="sigmoid", name="nsr")(output_nsr)

    if training_type == "all":
        outputs.append(output_mlp)
        outputs.append(output_nsr)
    elif training_type == "mlp":
        outputs.append(output_mlp)
    elif training_type == "nsr":
        outputs.append(output_nsr)

    return Model(inputs=bert_model.inputs, outputs=outputs)


def mlp_loss(y_true, y_pred):
    max_args = argmax(y_true)
    mask = cast(not_equal(max_args, zeros_like(max_args)), dtype='float32')
    loss = switch(mask, categorical_crossentropy(y_true, y_pred, from_logits=True), zeros_like(mask, dtype=floatx()))
    return sum(loss) / (cast(sum(mask), dtype='float32') + epsilon())


def mlp_accuracy(y_true, y_pred):
    print(y_true)
    max_args = argmax(y_true)
    mask = cast(not_equal(max_args, zeros_like(max_args)), dtype='float32')
    points = switch(mask, cast(equal(argmax(y_true, -1), argmax(y_pred, -1)), dtype='float32'),
                    zeros_like(mask, dtype=floatx()))
    return sum(points) / cast(sum(mask), dtype='float32')
