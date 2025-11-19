import jax
import numpy as np

from .activation import deriv_sigmoid, sigmoid
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
        self.init_layer = Layer((inputs.shape[1]), width, 1, activation)
        self.hidden_layer = [
            Layer(width, width, (i + 2), activation) for i in range(depth - 1)
        ]
        self.output_layer = Layer(width, 1, (depth + 1), activation)
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

    def _collect_params(self):
        """Function that collects all the weights and biases of every neuron"""

        params_dict = {}

        for layer in self.network:
            # Handle hidden layer list separately
            if isinstance(layer, list):
                for hidden_layer in layer:
                    layer_id = f"Layer_{hidden_layer.id}"
                    params_dict[layer_id] = {}

                    for neuron in hidden_layer.neurons:
                        neuron_id = f"Neuron_{neuron.id}"
                        params_dict[layer_id][neuron_id] = {
                            "weights": neuron.weights,
                            "bias": neuron.bias,
                            "z": getattr(neuron, "z", None),
                        }
            else:
                # Handle regular layers (init and output)
                layer_id = f"Layer_{layer.id}"
                params_dict[layer_id] = {}

                for neuron in layer.neurons:
                    neuron_id = f"Neuron_{neuron.id}"
                    params_dict[layer_id][neuron_id] = {
                        "weights": neuron.weights,
                        "bias": neuron.bias,
                        "z": getattr(neuron, "z", None),
                    }

        return params_dict

    def _backprop(self, x, y_true):
        """Compute deltas for backpropagation."""

        # Forward pass to get predictions
        y_hat = self.feedforward(x)

        # Collect all parameters (including z values)
        params = self._collect_params()

        # Gradient of loss with respect to output
        dl_dyhat = deriv_mse_loss(y_true, y_hat)

        # Initialize deltas dictionary
        deltas = {}

        # Calculate output layer delta
        output_layer_id = self.depth + 1
        output_layer_key = f"Layer_{output_layer_id}"

        if output_layer_key in params:
            # Get z values for output layer neurons
            output_z_values = []
            for neuron_key in params[output_layer_key]:
                neuron_z = params[output_layer_key][neuron_key]["z"]
                output_z_values.append(neuron_z)

            # Convert to array and calculate delta
            z_array = np.array(output_z_values).flatten()
            delta_output = dl_dyhat.flatten() * deriv_sigmoid(z_array)
            deltas[output_layer_key] = delta_output

        # Calculate hidden layer deltas (backward through layers)
        for layer_id in range(self.depth, 0, -1):
            layer_key = f"Layer_{layer_id}"
            next_layer_key = f"Layer_{layer_id + 1}"

            if layer_key in params and next_layer_key in deltas:
                # Get weights from next layer
                next_layer_weights = []
                for neuron_key in params[next_layer_key]:
                    weights = params[next_layer_key][neuron_key]["weights"]
                    next_layer_weights.append(weights.flatten())

                # Create weight matrix (each row is a neuron's weights)
                W_next = np.array(next_layer_weights)

                # Get z values for current layer
                current_z_values = []
                for neuron_key in params[layer_key]:
                    neuron_z = params[layer_key][neuron_key]["z"]
                    current_z_values.append(neuron_z)

                z_array = np.array(current_z_values).flatten()

                # Calculate delta: W_next.T @ delta_next * sigmoid'(z)
                delta_current = (
                    W_next.T @ deltas[next_layer_key] * deriv_sigmoid(z_array)
                )
                deltas[layer_key] = delta_current

        pass

    def train(self, x, y, epochs, batch_size, learning_rate):
        """Train the network using given training data for a number of epochs."""

        for epoch in range(epochs):
            batches = self._split_batch(batch_size, x)
            for batch in batches:
                pass
