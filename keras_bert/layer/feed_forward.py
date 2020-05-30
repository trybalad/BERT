from tensorflow.keras.layers import Layer, Dense


class FeedForward(Layer):
    def __init__(self, nodes_num, feed_forward_depth, **kwargs):
        super().__init__(**kwargs)
        self.in_layer = Dense(feed_forward_depth, activation='relu')
        self.out_layer = Dense(nodes_num)

    def __call__(self, layer_input, **kwargs):
        val = self.in_layer(layer_input)
        return self.out_layer(val)
