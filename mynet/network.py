import matplotlib.pyplot as plt
import numpy as np

from .activation import deriv_sigmoid, sigmoid
from .layer import Layer
from .loss import deriv_mse_loss, mse_loss
from .network_dashboard import _loss_table_view, _mini_dashboard, _simple_terminal_graph


class Network:
    """A neural network with:
    - n inputs
    - h hidden layers
    - y outputs"""

    def __init__(self, inputs, depth, width, activation=sigmoid):
        """Initialize the Network with given architecture and parameters."""

        self.inputs = inputs
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

    def _get_z(self, params, layer_key):
        current_z_values = []
        for neuron_key in params[layer_key]:
            neuron_z = params[layer_key][neuron_key]["z"]
            current_z_values.append(neuron_z)

        return np.array(current_z_values).flatten()

    def _backprop(self, params, y_true, y_hat):
        """Compute deltas for backpropagation."""

        # Gradient of loss with respect to output
        dl_dyhat = deriv_mse_loss(y_true, y_hat)
        batch_size = y_true.shape[0]

        # Initialize deltas dictionary
        deltas = {}

        # Calculate output layer delta
        output_layer_id = self.depth + 1
        output_layer_key = f"Layer_{output_layer_id}"

        if output_layer_key in params:
            # Get z values for output layer neurons
            z_array = self._get_z(params, output_layer_key)
            # Ensure shapes match for element-wise multiplication
            if dl_dyhat.shape != z_array.shape:
                dl_dyhat = dl_dyhat.reshape(z_array.shape)
            delta_output = dl_dyhat * deriv_sigmoid(z_array)
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

                # Get z values and deltas for current layer
                z_array = self._get_z(params, layer_key)
                next_deltas = deltas[next_layer_key]

                # Get number of neurons in current and next layers
                current_layer_neurons = len(params[layer_key])
                next_layer_neurons = len(params[next_layer_key])

                # Reshape z_array to (batch_size, current_layer_neurons)
                z_reshaped = z_array.reshape(batch_size, current_layer_neurons)

                # Reshape next_deltas to (batch_size, next_layer_neurons)
                if len(next_deltas.shape) == 1:
                    next_deltas_reshaped = next_deltas.reshape(
                        batch_size, next_layer_neurons
                    )
                else:
                    next_deltas_reshaped = next_deltas

                # Calculate delta: delta_next @ W_next * sigmoid'(z)
                # Shape: (batch_size, next_layer_neurons) @ (next_layer_neurons, current_layer_neurons)
                #      = (batch_size, current_layer_neurons)
                delta_current = (next_deltas_reshaped @ W_next) * deriv_sigmoid(
                    z_reshaped
                )

                # Flatten back to the expected format
                delta_current = delta_current.flatten()

                deltas[layer_key] = delta_current

        return deltas

    def update_values(
        self, x: np.array, params: dict, deltas: dict, learning_rate: float
    ) -> dict:
        """updates parameters: weight and bias"""

        batch_size = x.shape[0]

        for layers in range(1, (self.depth + 2)):
            layer_key = f"Layer_{layers}"
            layer = params[layer_key]

            # first layer
            if layers == 1:
                a = x.copy()  # Shape: (batch_size, input_features)
                delta = deltas[layer_key]  # Shape: (batch_size * num_neurons,)

                # Reshape deltas to (batch_size, num_neurons)
                num_neurons = len(layer)
                delta_reshaped = delta.reshape(batch_size, num_neurons)

                # change values
                i = 0
                for neuron_id, neuron in layer.items():
                    # Get deltas for this specific neuron across all batch samples
                    neuron_deltas = delta_reshaped[:, i]  # Shape: (batch_size,)

                    # Calculate gradients averaged over the batch
                    # dw = (1/batch_size) * sum(delta_i * x_i) for each sample i
                    dw = (neuron_deltas.reshape(-1, 1) * a).mean(axis=0).reshape(-1, 1)
                    db = neuron_deltas.mean()  # Average bias gradient

                    # Update parameters
                    neuron["weights"] -= learning_rate * dw
                    neuron["bias"] -= learning_rate * db

                    i += 1

            else:
                # Get activations from previous layer (sigmoid of z values)
                prev_layer_key = f"Layer_{layers-1}"
                if prev_layer_key in params:
                    prev_z = self._get_z(params, prev_layer_key)
                    # Reshape to (batch_size, num_prev_neurons)
                    prev_neurons_count = len(params[prev_layer_key])
                    prev_z_reshaped = prev_z.reshape(batch_size, prev_neurons_count)
                    a = sigmoid(
                        prev_z_reshaped
                    )  # Shape: (batch_size, num_prev_neurons)
                else:
                    continue  # Skip if previous layer not found

                delta = deltas[layer_key]  # Shape: (batch_size * num_neurons,)

                # Reshape deltas to (batch_size, num_neurons)
                num_neurons = len(layer)
                delta_reshaped = delta.reshape(batch_size, num_neurons)

                # change values
                i = 0
                for neuron_id, neuron in layer.items():
                    # Get deltas for this specific neuron across all batch samples
                    neuron_deltas = delta_reshaped[:, i]  # Shape: (batch_size,)

                    # Calculate gradients averaged over the batch
                    # dw = (1/batch_size) * sum(delta_i * a_i) for each sample i
                    dw = (neuron_deltas.reshape(-1, 1) * a).mean(axis=0).reshape(-1, 1)
                    db = neuron_deltas.mean()  # Average bias gradient

                    # Update parameters
                    neuron["weights"] -= learning_rate * dw
                    neuron["bias"] -= learning_rate * db

                    i += 1

        return params  # Return after processing all layers

    def _update_params(self, params):
        """Update network parameters from params dictionary"""
        for layer in self.network:
            # Handle hidden layer list separately
            if isinstance(layer, list):
                for hidden_layer in layer:
                    layer_key = f"Layer_{hidden_layer.id}"
                    if layer_key in params:
                        for neuron in hidden_layer.neurons:
                            neuron_key = f"Neuron_{neuron.id}"
                            if neuron_key in params[layer_key]:
                                neuron.weights = params[layer_key][neuron_key][
                                    "weights"
                                ]
                                neuron.bias = params[layer_key][neuron_key]["bias"]

            else:
                # Handle regular layers (init and output)
                layer_key = f"Layer_{layer.id}"
                if layer_key in params:
                    for neuron in layer.neurons:
                        neuron_key = f"Neuron_{neuron.id}"
                        if neuron_key in params[layer_key]:
                            neuron.weights = params[layer_key][neuron_key]["weights"]
                            neuron.bias = params[layer_key][neuron_key]["bias"]

        return None

    def _plot_loss(self, loss_dict):
        """Display training loss visualization in terminal"""
        if not loss_dict:
            print("No loss data to display")
            return

        epochs = list(loss_dict.keys())
        losses = list(loss_dict.values())

        _mini_dashboard(epochs, losses)
        _simple_terminal_graph(epochs, losses)
        _loss_table_view(epochs, losses)

    def train(self, x, y, epochs, batch_size, learning_rate):
        """Train the network using given training data for a number of epochs."""

        loss_cal = {}

        for epoch in range(epochs):
            batches = self._split_batch(batch_size, x)
            print(f"starting epoch: {epoch}")
            for batch in batches:
                # Forward pass to get predictions
                y_hat = self.feedforward(x)

                # Collect all parameters (including z values)
                params = self._collect_params()

                deltas = self._backprop(params, y, y_hat)

                # update values

                params = self.update_values(x, params, deltas, learning_rate)

                self._update_params(params)

            if epoch not in loss_cal:
                loss_cal[f"{epoch}"] = self._compute_loss(y, y_hat)
            continue

        self._plot_loss(loss_cal)
        return None
