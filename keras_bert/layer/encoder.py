from tensorflow.keras.layers import Dropout, LayerNormalization, Layer

from keras_bert.layer.feed_forward import FeedForward
from keras_bert.layer.multi_head_self_attention import MultiHeadSelfAttention


class Encoder(Layer):
    def __init__(self, model_depth, num_heads, feed_forward_depth, drop_rate=0.1):
        super().__init__()

        self.multi_head_attention = MultiHeadSelfAttention(model_depth, num_heads)
        self.feed_forward = FeedForward(model_depth, feed_forward_depth)

        self.attention_norm = LayerNormalization(epsilon=1e-6)
        self.ff_norm = LayerNormalization(epsilon=1e-6)

        self.attention_drop = Dropout(drop_rate)
        self.ff_drop = Dropout(drop_rate)

    def call(self, x, mask):
        attn_output = self.multi_head_attention(x, mask)
        attn_output = self.attention_drop(attn_output)

        out1 = self.attention_norm(x + attn_output)

        ff_output = self.feed_forward(out1)
        ff_output = self.ff_drop(ff_output)

        out2 = self.ff_norm(out1 + ff_output)

        return out2
