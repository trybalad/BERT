from tensorflow.keras.layers import Dropout, LayerNormalization, Layer

from keras_bert.layer.feed_forward import PointWiseFeedForward
from keras_bert.layer.multi_head_self_attention import MultiHeadSelfAttention

"""
Encoder Layer class.
It contains MultiHeadAttentionLayer And Point Wise Feed Forward Layer
Each layer have added Normalization and Dropout layer
:param embedding_dim: size of embedded data, number of nodes to be used
:param heads_num: number of heads in MultiHeadAttention layer
:param ff_dim: number of point wise feed forward dimensions
:param drop_rate: used drop rate for layers
:param norm_eps: epsilon for Normalization layers
"""


class Encoder(Layer):
    def __init__(self, embedding_dim, heads_num, ff_dim, drop_rate=0.1, norm_eps=1e-6):
        super().__init__()

        self.multi_head_attention = MultiHeadSelfAttention(embedding_dim, heads_num)
        self.feed_forward = PointWiseFeedForward(embedding_dim, ff_dim)

        self.attention_norm = LayerNormalization(epsilon=norm_eps)
        self.ff_norm = LayerNormalization(epsilon=norm_eps)

        self.attention_drop = Dropout(drop_rate)
        self.ff_drop = Dropout(drop_rate)

    def __call__(self, x):
        attn_output = self.multi_head_attention(x)
        attn_output = self.attention_drop(attn_output)

        out1 = self.attention_norm(x + attn_output)

        ff_output = self.feed_forward(out1)
        ff_output = self.ff_drop(ff_output)

        out2 = self.ff_norm(out1 + ff_output)

        return out2
