from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Add, Dense, Flatten
from tensorflow.keras.models import Model

from keras_bert.layer.encoder import Encoder

"""
Creates BERT model.
:param sequence_len: maximal length of input sequence
"""


def create_model(vocab_size: int, input_dim: int, model_depth: int, encoders_num: int, heads_num: int,
                 feed_forward_depth: int) -> Model:
    # Create input layers
    tokens = Input(shape=input_dim, name='Tokens')
    segments = Input(shape=input_dim, name='Segments')
    mask = Input(shape=input_dim, name='Mask')

    # Create embedding layers
    embeddings = create_embeddings_layer(tokens, segments, mask, vocab_size, input_dim, model_depth)

    # Create encoders
    encoder = create_encoder_layers(encoders_num, embeddings, model_depth, heads_num, feed_forward_depth)

    # return Model(inputs=[tokens, segments, mask], outputs=[encoder])
    output_layer = Dense(vocab_size)(encoder)
    return Model(inputs=[tokens, segments, mask], outputs=[output_layer])


def create_encoder_layers(encoders_num, embeddings, model_depth, heads_num, feed_forward_depth):
    encoder = Encoder(model_depth, heads_num, feed_forward_depth)(embeddings, None)
    for i in range(1, encoders_num):
        encoder = Encoder(model_depth, heads_num, feed_forward_depth)(encoder, None)
    return encoder


def create_embeddings_layer(tokens, segments, mask, vocab_size, input_dim, model_depth):
    tokens_emb = Embedding(vocab_size, model_depth, input_length=input_dim, name='Tokens_Embedding')(tokens)
    segments_emb = Embedding(2, model_depth, input_length=input_dim, name='Segments_Embedding')(segments)
    mask_emb = Embedding(input_dim, model_depth, input_length=input_dim, name='Mask_Embedding')(mask)
    return Add(name="Embeddings_sum")([tokens_emb, segments_emb, mask_emb])
