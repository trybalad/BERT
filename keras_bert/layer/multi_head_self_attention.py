from tensorflow import shape, transpose, reshape, matmul, cast, float32, math
from tensorflow.nn import softmax
from tensorflow.keras.layers import Layer, Dense


class MultiHeadSelfAttention(Layer):

    def __init__(self, num_nodes, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.num_nodes = num_nodes

        assert num_nodes % self.num_heads == 0

        self.depth = self.num_nodes // self.num_heads

        # queries
        self.wq = Dense(num_nodes)
        # keys
        self.wk = Dense(num_nodes)
        # values
        self.wv = Dense(num_nodes)

        self.final_linear = Dense(num_nodes)

    def __call__(self, layer_input, mask):
        batch_size = shape(layer_input)[0]

        # input values go through dense layers
        q = self.wq(layer_input)
        k = self.wk(layer_input)
        v = self.wv(layer_input)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = reshape(scaled_attention, (batch_size, -1, self.num_nodes))

        output = self.final_linear(concat_attention)

        return output

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
          q: query shape == (..., seq_len_q, depth)
          k: key shape == (..., seq_len_k, depth)
          v: value shape == (..., seq_len_v, depth_v)
          mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
          output, attention_weights
        """

        matmul_qk = matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

        # scale matmul_qk
        dk = cast(shape(k)[-1], float32)
        scaled_attention_logits = matmul_qk / math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

            # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        output = matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights
