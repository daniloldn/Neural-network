import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.activation import sigmoid
from neural_network.network import Network


def test_network_single_output_shape():
    """Test that Network returns proper array shape for single output."""
    # Single observation with 2 features
    x = np.array([[1.0, 2.0]])
    # dummy target (not used in feedforward but needed for constructor)
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=2)
    output = net.feedforward(x)

    # Should return array with shape (1, 1) for single observation
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 1)  # 1 row, 1 output


def test_network_multivariate_output_shape():
    """Test Network with multiple output neurons."""
    x = np.array([[1.0, 2.0]])  # Single observation
    y = np.array([[0.0, 0.0]])  # 2 outputs expected

    # Create network with 2 output neurons by modifying after construction
    net = Network(x, y, depth=1, width=3)
    # Replace output layer to have 2 neurons for multivariate output
    from neural_network.layer import Layer

    net.output_layer = Layer(3, 2, activation=sigmoid)

    output = net.feedforward(x)

    # Should return array with shape (1, 2) for single observation, 2 outputs
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 2)  # 1 row, 2 outputs


def test_network_single_row_input():
    """Test Network with single row input."""
    x = np.array([[1.0, 2.0, 3.0]])  # Single observation
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=2)
    output = net.feedforward(x)

    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 1)  # 1 row, 1 output


def test_network_output_values_reasonable():
    """Test that network outputs are in reasonable range for sigmoid activation."""
    x = np.array([[0.0, 0.0]])  # Single observation
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=2, activation=sigmoid)
    output = net.feedforward(x)

    # Sigmoid outputs should be in range (0, 1)
    assert np.all(output > 0.0)
    assert np.all(output < 1.0)
    assert output.shape == (1, 1)


def test_network_deterministic_weights():
    """Test network with deterministic weights for reproducible results."""
    # Seed for reproducible results
    np.random.seed(42)

    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    y = np.array([[0.0], [0.0]])

    net = Network(x, y, depth=1, width=2, activation=lambda z: z)  # linear activation

    # Set deterministic weights
    net.init_layer.neurons[0].weights = np.array([[1.0], [0.0]])
    net.init_layer.neurons[0].bias = 0.0
    net.init_layer.neurons[1].weights = np.array([[0.0], [1.0]])
    net.init_layer.neurons[1].bias = 0.0

    net.output_layer.neurons[0].weights = np.array([[1.0], [1.0]])
    net.output_layer.neurons[0].bias = 0.0

    output = net.feedforward(x)

    # First input [1,0] -> init_layer -> [1,0] -> output -> 1
    # Second input [0,1] -> init_layer -> [0,1] -> output -> 1
    expected = np.array([[1.0], [1.0]])
    assert np.allclose(output, expected)


def test_network_different_depths():
    """Test networks with different depths work correctly."""
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    # Depth 1 (no hidden layers)
    net1 = Network(x, y, depth=1, width=2)
    output1 = net1.feedforward(x)
    assert output1.shape == (1, 1)

    # Depth 2 (one hidden layer)
    net2 = Network(x, y, depth=2, width=2)
    output2 = net2.feedforward(x)
    assert output2.shape == (1, 1)

    # Depth 3 (two hidden layers)
    net3 = Network(x, y, depth=3, width=2)
    output3 = net3.feedforward(x)
    assert output3.shape == (1, 1)


def test_network_different_input_dimensions():
    """Test network with different input feature dimensions."""
    # 1 feature
    x1 = np.array([[1.0]])  # Single observation, 1 feature
    y1 = np.array([[0.0]])
    net1 = Network(x1, y1, depth=1, width=2)
    output1 = net1.feedforward(x1)
    assert output1.shape == (1, 1)

    # 5 features
    x5 = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])  # Single observation, 5 features
    y5 = np.array([[0.0]])
    net5 = Network(x5, y5, depth=1, width=3)
    output5 = net5.feedforward(x5)
    assert output5.shape == (1, 1)


def test_network_consistency():
    """Test that same input produces same output (deterministic)."""
    np.random.seed(123)

    x = np.array([[1.0, 2.0]])  # Single observation
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=2)

    # Run feedforward twice with same input
    output1 = net.feedforward(x)
    output2 = net.feedforward(x)

    # Should get identical results
    assert np.array_equal(output1, output2)
    assert output1.shape == output2.shape == (1, 1)
