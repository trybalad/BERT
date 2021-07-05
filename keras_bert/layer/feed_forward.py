from tensorflow.keras.layers import Layer, Dense

"""
Point-Wise Feed-Forward Layer class.
Creates fully connected feed forward layer with `relu` activation in between.
:param output_nodes_num: input and output size of nodes
:param feed_forward_dimensions: number of nodes in the inner-layer 
"""


class PointWiseFeedForward(Layer):
    def __init__(self, output_nodes_num, feed_forward_dimensions):
        super().__init__()
        self.in_layer = Dense(feed_forward_dimensions, activation='relu')
        self.out_layer = Dense(output_nodes_num)

    def __call__(self, layer_input):
        val = self.in_layer(layer_input)
        return self.out_layer(val)
