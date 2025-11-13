import numpy as np

from .activation import sigmoid
from .layer import Layer
from .loss import deriv_mse_loss, mse_loss


class Network:
    """A neural network with:
    - n inputs
    - h hidden layers
    - y outputs"""

    def __init__(self, inputs, depth, width, activation=sigmoid):
        """Initialize the Network with given architecture and parameters."""

        self.depth = depth
        self.init_layer = Layer((inputs.shape[1]), width, activation)
        self.hidden_layer = [Layer(width, width, activation) for _ in range(depth - 1)]
        self.output_layer = Layer(width, 1, activation)
        self.network = [self.init_layer, self.hidden_layer, self.output_layer]

    def _hidden_recursion(self, x):
        """Recursively feed forward through hidden layers"""

        def _recursive_forward(layer_input, layer_index):
            """Helper function to recursively feed forward through hidden layers."""
            if layer_index >= len(self.hidden_layer):
                return layer_input

            current_output = self.hidden_layer[layer_index].feed_layer(layer_input)
            return _recursive_forward(current_output, layer_index + 1)

        return _recursive_forward(x, 0)

    def feedforward(self, x):
        """Feed input x through the network and return output."""

        first_step = self.init_layer.feed_layer(x)
        if self.depth > 1:
            hidden_step = self._hidden_recursion(first_step)
            output = self.output_layer.feed_layer(hidden_step)
        else:
            output = self.output_layer.feed_layer(first_step)
        return output

    def _compute_loss(self, y_true, y_pred):
        return mse_loss(y_true, y_pred)

    def _split_batch(self, batch_size, x):
        shuffled_x = x.copy()
        np.random.shuffle(shuffled_x)
        n_batch = max(1, round(x.shape[0] / batch_size))
        return np.array_split(shuffled_x, n_batch)

    def _deriv_output(self, out_neuron):
        bias = out_neuron.neurons.bias

    def train(self, x, y, epochs, batch_size, learning_rate):
        """Train the network using given training data for a number of epochs."""

        for epoch in range(epochs):
            batches = self._split_batch(batch_size, x)
            for batch in batches:
                pass
