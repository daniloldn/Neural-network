import sys
from pathlib import Path

import numpy as np

# Tests run under pytest may not have the repo root on sys.path. Mirror
# the project's quick scripts and add the parent directory so imports work
# when pytest is invoked from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.activation import sigmoid
from neural_network.layer import Layer
from neural_network.network import Network
from neural_network.neuron import Neuron


def test_neuron_math():
    # deterministic weights and bias
    n = Neuron(2, 0, activation=sigmoid)
    n.weights = np.array([[0.5], [-1.0]])
    n.bias = 0.1

    x = np.array([2.0, 3.0])

    # explicit compute
    z = np.dot(x, n.weights) + n.bias
    expected = sigmoid(z)

    out = n.feed_forward(x)

    assert np.allclose(np.squeeze(out), np.squeeze(expected))


def test_layer_math_identity():
    # use identity activation so outputs equal dot-product + bias
    identity = lambda x: x
    l = Layer(2, 2, activation=identity)

    # neuron0 selects first input, neuron1 selects second input
    l.neurons[0].weights = np.array([[1.0], [0.0]])
    l.neurons[0].bias = 0.0
    l.neurons[1].weights = np.array([[0.0], [1.0]])
    l.neurons[1].bias = 0.0

    x = np.array([2.0, 3.0])
    out = l.feed_forward(x)

    # layer returns a list of per-neuron outputs (arrays). Squeeze to scalars.
    vals = np.array([np.squeeze(o) for o in out])
    assert np.allclose(vals, np.array([2.0, 3.0]))


def test_network_feedforward_row_sum_identity():
    # Build a network where the output is the row-wise sum of inputs.
    identity = lambda x: x
    x = np.array([[2.0, 3.0]])  # single-row input so shapes are simpler
    y = np.array([0.0])

    net = Network(x, y, depth=1, width=2, activation=identity)

    # init layer: neurons return x1 and x2 respectively
    net.init_layer.neurons[0].weights = np.array([[1.0], [0.0]])
    net.init_layer.neurons[0].bias = 0.0
    net.init_layer.neurons[1].weights = np.array([[0.0], [1.0]])
    net.init_layer.neurons[1].bias = 0.0

    # output layer: single neuron sums the two values
    net.output_layer.neurons[0].weights = np.array([[1.0], [1.0]])
    net.output_layer.neurons[0].bias = 0.0

    out = net.feedforward(x)

    # output is either a list containing an array or an array; normalize then squeeze
    if isinstance(out, list):
        final = out[0]
    else:
        final = out

    final_val = np.squeeze(final)
    assert np.allclose(final_val, 5.0)
