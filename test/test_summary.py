"""
Summary of Neural Network Tests

This file demonstrates that the neural network package works correctly 
for 1D inputs (single observations) and returns predicted arrays as expected.
"""

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.activation import sigmoid
from neural_network.network import Network


def test_network_returns_predicted_array():
    """Verify network returns an array of predictions for given inputs."""

    print("=== Neural Network Functionality Test ===\n")

    # Test 1: Basic functionality
    print("1. Testing basic network with single observation...")
    x = np.array([[1.0, 2.0]])  # Single observation, 2 features
    y = np.array([[0.0]])  # Target (not used in prediction)

    net = Network(x, y, depth=1, width=3)
    output = net.feedforward(x)

    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output value: {output[0,0]:.4f}")
    assert output.shape == (1, 1)
    assert isinstance(output, np.ndarray)
    print("   ✓ PASSED\n")

    # Test 2: Different input dimensions
    print("2. Testing with different input feature dimensions...")
    test_cases = [(1, "1 feature"), (3, "3 features"), (5, "5 features")]

    for n_features, desc in test_cases:
        x_test = np.array([np.random.randn(n_features)])  # Single observation
        y_test = np.array([[0.0]])

        net_test = Network(x_test, y_test, depth=1, width=4)
        output_test = net_test.feedforward(x_test)

        print(f"   {desc}: Input {x_test.shape} -> Output {output_test.shape}")
        assert output_test.shape == (1, 1)
    print("   ✓ PASSED\n")

    # Test 3: Different network depths
    print("3. Testing with different network depths...")
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0]])

    for depth in [1, 2, 3]:
        net_depth = Network(x, y, depth=depth, width=3)
        output_depth = net_depth.feedforward(x)

        print(
            f"   Depth {depth}: Output {output_depth.shape}, Value: {output_depth[0,0]:.4f}"
        )
        assert output_depth.shape == (1, 1)
    print("   ✓ PASSED\n")

    # Test 4: Multivariate output (modifying network structure)
    print("4. Testing multivariate output...")
    x = np.array([[1.0, 2.0]])
    y = np.array([[0.0, 0.0]])  # 2 outputs expected

    from neural_network.layer import Layer

    net_multi = Network(x, y, depth=1, width=3)
    # Replace output layer to have 2 neurons
    net_multi.output_layer = Layer(3, 2, activation=sigmoid)

    output_multi = net_multi.feedforward(x)
    print(f"   Multivariate output shape: {output_multi.shape}")
    print(f"   Output values: [{output_multi[0,0]:.4f}, {output_multi[0,1]:.4f}]")
    assert output_multi.shape == (1, 2)
    print("   ✓ PASSED\n")

    # Test 5: Numerical stability
    print("5. Testing numerical stability with extreme values...")
    x_extreme = np.array([[100.0, -100.0]])  # Very large values
    net_stable = Network(
        x_extreme, np.array([[0.0]]), depth=1, width=3, activation=sigmoid
    )
    output_stable = net_stable.feedforward(x_extreme)

    print(f"   Extreme input: {x_extreme[0]}")
    print(f"   Sigmoid output: {output_stable[0,0]:.6f} (should be between 0 and 1)")
    assert 0 < output_stable[0, 0] < 1
    assert not np.isnan(output_stable).any()
    print("   ✓ PASSED\n")

    print("🎉 All tests passed! Neural network is working correctly.")
    print("✅ Network successfully returns predicted arrays for 1D inputs.")


if __name__ == "__main__":
    test_network_returns_predicted_array()
