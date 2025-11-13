import numpy as np

from .activation import sigmoid
from .layer import Layer


class Network:
    """A neural network with:
    - n inputs
    - h hidden layers
    - y outputs"""

    def __init__(self, inputs, output, depth, width, activation=sigmoid, loss=None):

        self.loss = loss
        self.depth = depth
        self.init_layer = Layer((inputs.shape[1]), width, activation)
        self.hidden_layer = [Layer(width, width, activation) for _ in range(depth - 1)]
        self.output_layer = Layer(width, 1, activation, final=True)
        self.network = [self.init_layer, self.hidden_layer, self.output_layer]

    def _hidden_recursion(self, x):
        """Recursively feed forward through hidden layers"""
        print(x, "x in hidden recursion")
        print(len(self.hidden_layer), "length of hidden layer")

        def _recursive_forward(layer_input, layer_index):
            print(layer_index, "layer index")
            if layer_index >= len(self.hidden_layer):
                return layer_input

            print("calling feed layer for hidden layer index:", layer_index)
            current_output = self.hidden_layer[layer_index].feed_layer(layer_input)
            return _recursive_forward(current_output, layer_index + 1)

        return _recursive_forward(x, 0)

    def feedforward(self, x):
        print("calling first step with x:", x)
        first_step = self.init_layer.feed_layer(x)
        print(first_step, "first step")
        if self.depth > 1:
            print("we are in the hiddden layers")
            hidden_step = self._hidden_recursion(first_step)
            print("calling output layer with hidden step:", hidden_step)
            output = self.output_layer.feed_layer(hidden_step)
        else:
            output = self.output_layer.feed_layer(first_step)
        return output
