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
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embedding_dim} should be divisible by number of heads = {num_heads}"
            )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.projection_dim = embedding_dim // num_heads
        self.query_dense = Dense(embedding_dim)
        self.key_dense = Dense(embedding_dim)
        self.value_dense = Dense(embedding_dim)
        self.combine_heads = Dense(embedding_dim)

    @staticmethod
    def attention(query, key, value):
        score = matmul(query, key, transpose_b=True)
        dim_key = cast(shape(key)[-1], float32)
        scaled_score = score / math.sqrt(dim_key)
        weights = softmax(scaled_score, axis=-1)
        output = matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return transpose(x, perm=[0, 2, 1, 3])

    def __call__(self, inputs, mask):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = reshape(
            attention, (batch_size, -1, self.embedding_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

