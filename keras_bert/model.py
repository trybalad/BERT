from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Add, Dense, Flatten
from tensorflow.keras.models import Model

from keras_bert.layer.encoder import Encoder

"""
Creates BERT model.
:param vocab_size: size of used vocabulary
:param input_dim: dimensions of input tokens (max_len when creating input)
:param embedding_dim: dimensions after input embeddings
:param encoders_num: number of encoder layers
:param heads_num: number of heads in MultiHeadAttention layer
:param ff_dim: number of point wise feed forward dimensions
"""


def create_model(vocab_size: int, input_dim: int, embedding_dim: int, encoders_num: int, heads_num: int,
                 ff_dim: int) -> Model:
    # Create input layers
    tokens = Input(shape=input_dim, name='Tokens')
    segments = Input(shape=input_dim, name='Segments')
    mask = Input(shape=input_dim, name='Mask')

    """Create embedding layers"""
    embeddings = create_embeddings_layer(tokens, segments, mask, vocab_size, input_dim, embedding_dim)

    """Create encoders"""
    encoder = create_encoder_layers(encoders_num, embeddings, embedding_dim, heads_num, ff_dim)

    """Create model"""
    return Model(inputs=[tokens, segments, mask], outputs=[encoder])


def create_encoder_layers(encoders_num, embeddings, nodes_num, heads_num, ff_dim):
    """
    Creates encoders_num of Encoder layers, starting with layer with output from embeddings.
    Number of nodes and number of heads and dimensions of point wise feed forward network are given.
    """
    encoder = Encoder(nodes_num, heads_num, ff_dim)(embeddings, None)
    for i in range(1, encoders_num):
        encoder = Encoder(nodes_num, heads_num, ff_dim)(encoder, None)
    return encoder


def create_embeddings_layer(tokens, segments, mask, vocab_size, input_dim, embedding_size):
    """
    Creates 3 Embeddings layers for Tokens, Segments and Masks.
    And then adds it into single tensor
    TODO Check if own implementation of Embedding layer is not needed
    """
    tokens_emb = Embedding(vocab_size, embedding_size, input_length=input_dim, name='Tokens_Embedding')(tokens)
    segments_emb = Embedding(2, embedding_size, input_length=input_dim, name='Segments_Embedding')(segments)
    mask_emb = Embedding(input_dim, embedding_size, input_length=input_dim, name='Mask_Embedding')(mask)
    return Add(name="Embeddings_sum")([tokens_emb, segments_emb, mask_emb])
