from tensorflow import shape, transpose, reshape, matmul, cast, float32, math
from tensorflow.keras.layers import Layer, Dense
from tensorflow.nn import softmax

"""
Multi Head Self Attention Layer class.

:param embedding_dim: number of dimensions of embeded data
:param num_heads: number of heads 
"""


class MultiHeadSelfAttention(Layer):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.projection_dim = embedding_dim // num_heads

        self.check_correct_params()

        self.query_layer = Dense(embedding_dim)
        self.key_layer = Dense(embedding_dim)
        self.value_layer = Dense(embedding_dim)
        self.combine_heads = Dense(embedding_dim)

    def check_correct_params(self):
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(
                "Wrong parameters, number of embedding dimensions should be divisable by number of heads"
            )

    """ Calculates attention for given vectors"""
    @staticmethod
    def attention(query, key, value):
        score = matmul(query, key, transpose_b=True)
        dim_key = cast(shape(key)[-1], float32)
        scaled_score = score / math.sqrt(dim_key)
        weights = softmax(scaled_score, axis=-1)
        output = matmul(weights, value)
        return output

    """ Split word vectors into chunks of size num_heads, and divide work"""
    def separate_heads(self, x, batch_size):
        x = reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return transpose(x, perm=[0, 2, 1, 3])

    """ Layer use self-attention, so we are using one input for all layers"""
    def __call__(self, inputs):

        batch_size = shape(inputs)[0]

        """Pass input through all the base layers"""
        query = self.query_layer(inputs)
        key = self.key_layer(inputs)
        value = self.value_layer(inputs)

        """Splits vector representation of words into num_heads parts"""
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        """Calculate value of self-attention for parts"""
        attention = self.attention(query, key, value)
        attention = transpose(attention, perm=[0, 2, 1, 3])

        """Combine information learnt by different heads"""
        concat_attention = reshape(attention, (batch_size, -1, self.embedding_dim))
        output = self.combine_heads(concat_attention)
        return output

