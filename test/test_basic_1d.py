import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.network import Network


def test_basic_network_functionality():
    """Basic test to ensure network works with 1D inputs."""
    # Single observation, 2 features
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    net = Network(x, y, depth=1, width=3)
    output = net.feedforward(x)

    # Basic assertions
    assert isinstance(output, np.ndarray)
    assert output.shape == (1, 1)
    assert not np.isnan(output).any()
    assert not np.isinf(output).any()
    print("✓ Basic network test passed")


def test_network_with_different_widths():
    """Test network with different layer widths."""
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    for width in [1, 2, 5, 10]:
        net = Network(x, y, depth=1, width=width)
        output = net.feedforward(x)
        assert output.shape == (1, 1)

    print("✓ Different width test passed")


def test_network_with_different_depths():
    """Test network with different depths."""
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    for depth in [1, 2, 3]:
        net = Network(x, y, depth=depth, width=3)
        output = net.feedforward(x)
        assert output.shape == (1, 1)

    print("✓ Different depth test passed")


if __name__ == "__main__":
    test_basic_network_functionality()
    test_network_with_different_widths()
    test_network_with_different_depths()
    print("\n✅ All basic tests passed!")
