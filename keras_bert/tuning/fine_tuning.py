from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import categorical_crossentropy, binary_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, binary_accuracy
from tensorflow.keras.optimizers import Adam

from keras_bert.training import prepare_pretrain_model_from_checkpoint


def fine_tune(bert_model: Model, tokenizer, max_len: int, data_generator,
              epochs=10, checkpoint_file_path=None, load_checkpoint=False, old_checkpoint=None, learning_rate=2e-5,
              learn_type="all"):
    print("Max len of tokens:", max_len)

    tune_model = prepare_fine_tune_model_from_checkpoint(bert_model, tokenizer, checkpoint_file_path,
                                                         load_checkpoint,
                                                         old_checkpoint, learning_rate, learn_type)

    if checkpoint_file_path:
        checkpoint = [ModelCheckpoint(filepath=checkpoint_file_path, save_weights_only=True, verbose=1)]
    else:
        checkpoint = None

    print(tune_model.summary())
    tune_model.fit(data_generator, epochs=epochs,
                   callbacks=[checkpoint])
    return tune_model


def load_pretrained_model(bert_model: Model, checkpoint_file_path=None, task="all", learning_rate=2e-5):

    tune_model = prepare_fine_tuning_model(bert_model, task)
    loss, loss_weights, metrics = prepare_losses_and_metrics(task)
    tune_model.compile(optimizer=Adam(learning_rate), loss=loss, loss_weights=loss_weights, metrics=metrics)

    tune_model.load_weights(checkpoint_file_path)
    return tune_model


def prepare_fine_tune_model_from_checkpoint(bert_model: Model, tokenizer, checkpoint_file_path=None,
                                            load_checkpoint=True, old_checkpoint=None, learning_rate=2e-5,
                                            learn_type="all"):
    model = prepare_pretrain_model_from_checkpoint(bert_model, tokenizer, checkpoint_file_path=checkpoint_file_path,
                                                   load_checkpoint=load_checkpoint, old_checkpoint=old_checkpoint,
                                                   learning_rate=2e-5,
                                                   learn_type="all")

    pretrained = Model(model.inputs, model.layers[-1].input, name="pretrained_model")
    tune_model = prepare_fine_tuning_model(pretrained, learn_type)
    loss, loss_weights, metrics = prepare_losses_and_metrics(learn_type)
    tune_model.compile(optimizer=Adam(learning_rate), loss=loss, loss_weights=loss_weights, metrics=metrics)
    return tune_model


def prepare_fine_tuning_model(bert_model, training_type):
    outputs = []

    if training_type == "ar":
        output_ar = Flatten()(bert_model.layers[-1].input)
        output_ar = Dense(5, activation="softmax", name="ar")(output_ar)
        outputs.append(output_ar)

    elif training_type == "dyk":
        output_dyk = Flatten()(bert_model.layers[-1].input)
        output_dyk = Dense(1, activation="sigmoid", name="dyk")(output_dyk)
        outputs.append(output_dyk)

    elif training_type == "cbd":
        output_cbd = Flatten()(bert_model.layers[-1].input)
        output_cbd = Dense(1, activation="sigmoid", name="cbd")(output_cbd)
        outputs.append(output_cbd)

    elif training_type == "cdsc":
        output_cdsc = Flatten()(bert_model.layers[-1].input)
        output_cdsc = Dense(3, activation="softmax", name="cdsc")(output_cdsc)
        outputs.append(output_cdsc)

    return Model(inputs=bert_model.inputs, outputs=outputs)


def prepare_losses_and_metrics(learn_type):
    if learn_type == "ar":
        loss = categorical_crossentropy
        metrics = categorical_accuracy
        return loss, None, metrics
    elif learn_type == "dyk":
        loss = binary_crossentropy
        metrics = binary_accuracy
        return loss, None, metrics
    elif learn_type == "cbd":
        loss = binary_crossentropy
        metrics = binary_accuracy
        return loss, None, metrics
    elif learn_type == "cdsc":
        loss = categorical_crossentropy
        metrics = categorical_accuracy
        return loss, None, metrics
    else:
        return None
