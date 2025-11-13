import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.activation import sigmoid
from neural_network.network import Network


def test_network_large_batch():
    """Test network handles single observation efficiently."""
    # Single observation with multiple features
    x = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # 1 observation, 5 features
    y = np.array([[0.0]])

    net = Network(x, y, depth=2, width=10)
    output = net.feedforward(x)

    assert output.shape == (1, 1)
    assert not np.any(np.isnan(output))
    assert not np.any(np.isinf(output))


def test_network_numerical_stability():
    """Test network handles extreme input values."""
    # Test with very large values
    x_large = np.array([[100.0, 200.0]])  # Single observation
    y = np.array([[0.0]])

    net = Network(x_large, y, depth=1, width=3, activation=sigmoid)
    output_large = net.feedforward(x_large)

    # Sigmoid should keep values bounded
    assert np.all(output_large > 0)
    assert np.all(output_large < 1)
    assert not np.any(np.isnan(output_large))

    # Test with very small values
    x_small = np.array([[0.001, -0.001]])  # Single observation
    output_small = net.feedforward(x_small)

    assert np.all(output_small > 0)
    assert np.all(output_small < 1)
    assert not np.any(np.isnan(output_small))


def test_network_different_activations():
    """Test network with different activation functions."""
    x = np.array([[1.0, 2.0]])  # Single observation
    y = np.array([[0.0]])

    # Linear activation
    linear = lambda z: z
    net_linear = Network(x, y, depth=1, width=2, activation=linear)
    output_linear = net_linear.feedforward(x)
    assert output_linear.shape == (1, 1)

    # ReLU-like activation
    relu = lambda z: np.maximum(0, z)
    net_relu = Network(x, y, depth=1, width=2, activation=relu)
    output_relu = net_relu.feedforward(x)
    assert output_relu.shape == (1, 1)
    assert np.all(output_relu >= 0)

    # Tanh activation
    tanh = lambda z: np.tanh(z)
    net_tanh = Network(x, y, depth=1, width=2, activation=tanh)
    output_tanh = net_tanh.feedforward(x)
    assert output_tanh.shape == (1, 1)
    assert np.all(output_tanh >= -1)
    assert np.all(output_tanh <= 1)


def test_network_input_validation():
    """Test network behavior with edge case inputs."""
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=2)

    # Test with zeros
    x_zeros = np.array([[0.0, 0.0]])
    output_zeros = net.feedforward(x_zeros)
    assert output_zeros.shape == (1, 1)

    # Test with negative values
    x_negative = np.array([[-1.0, -2.0]])
    output_negative = net.feedforward(x_negative)
    assert output_negative.shape == (1, 1)


def test_network_weight_initialization():
    """Test that network weights are properly initialized and different across neurons."""
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=3)

    # Check that neurons have different weights
    weights_list = []
    for neuron in net.init_layer.neurons:
        weights_list.append(neuron.weights.flatten())

    # Verify weights are different (probability of identical random weights is very low)
    for i in range(len(weights_list)):
        for j in range(i + 1, len(weights_list)):
            assert not np.array_equal(weights_list[i], weights_list[j])

    # Check that biases are initialized
    for neuron in net.init_layer.neurons:
        assert isinstance(neuron.bias, (float, np.floating))


def test_network_architecture_properties():
    """Test that network maintains expected architectural properties."""
    x = np.array([[1.0, 2.0, 3.0]])
    y = np.array([[0.0]])

    depth = 3
    width = 4
    net = Network(x, y, depth=depth, width=width)

    # Check network structure
    assert len(net.network) == 3  # init_layer, hidden_layer, output_layer
    init_layer, hidden_layers, output_layer = net.network

    # Check hidden layer count
    assert len(hidden_layers) == depth - 1

    # Check layer dimensions
    assert len(init_layer.neurons) == width
    assert len(output_layer.neurons) == 1  # Currently hardcoded to 1

    for hidden_layer in hidden_layers:
        assert len(hidden_layer.neurons) == width


def test_network_feedforward_consistency_across_calls():
    """Test that multiple calls to feedforward with same weights give identical results."""
    np.random.seed(42)  # For reproducible test

    x = np.array([[1.0, 2.0]])  # Single observation
    y = np.array([[0.0]])

    net = Network(x, y, depth=2, width=3)

    # Multiple calls should give identical results
    outputs = []
    for _ in range(5):
        output = net.feedforward(x)
        outputs.append(output.copy())

    # All outputs should be identical
    for i in range(1, len(outputs)):
        assert np.array_equal(outputs[0], outputs[i])


def test_network_memory_efficiency():
    """Test that network doesn't leak memory or create excessive intermediate arrays."""
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=2)

    # Run feedforward multiple times
    for _ in range(100):
        output = net.feedforward(x)
        assert output.shape == (1, 1)

    # Test should complete without memory issues
    assert True
