from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Add
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


def create_model(vocab_size: int, input_dim: int, embedding_dim: int,
                 encoders_num: int, heads_num: int,
                 ff_dim: int) -> Model:
    # Create input layers
    tokens = Input(shape=input_dim, name='Tokens')
    segments = Input(shape=input_dim, name='Segments')
    mask = Input(shape=input_dim, name='Mask')

    # Create embedding layers
    embeddings = create_embeddings_layer(tokens, segments, mask, vocab_size, input_dim, embedding_dim)

    # Create encoders
    encoder = create_encoder_layers(encoders_num, embeddings, embedding_dim, heads_num, ff_dim)

    # Create model
    return Model(inputs=[tokens, segments, mask], outputs=[encoder])


# Creates encoders_num of Encoder layers, starting with layer with output from embeddings.
# Number of nodes and number of heads and dimensions of point wise feed forward network are given.
def create_encoder_layers(encoders_num, embeddings, nodes_num, heads_num, ff_dim):
    encoder = Encoder(nodes_num, heads_num, ff_dim)(embeddings)
    for i in range(1, encoders_num):
        encoder = Encoder(nodes_num, heads_num, ff_dim)(encoder)
    return encoder


# Creates 3 Embeddings layers for Tokens, Segments and Masks.
# And then adds it into single tensor
def create_embeddings_layer(tokens, segments, mask, vocab_size, input_dim, embedding_size):
    tokens_emb = Embedding(vocab_size, embedding_size, input_length=input_dim, name='Tokens_Embedding')(tokens)
    segments_emb = Embedding(3, embedding_size, input_length=input_dim, name='Segments_Embedding')(segments)
    mask_emb = Embedding(input_dim, embedding_size, input_length=input_dim, name='Mask_Embedding')(mask)
    return Add(name="Embeddings_sum")([tokens_emb, segments_emb, mask_emb])


def load_model(vocab_size: int, input_dim: int, embedding_dim: int, encoders_num: int, heads_num: int,
               ff_dim: int, model_checkpoint):
    model = create_model(vocab_size, input_dim, embedding_dim, encoders_num, heads_num, ff_dim)
    model.load_weights(model_checkpoint)
    return model
