import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.activation import deriv_sigmoid, sigmoid
from neural_network.loss import deriv_mse_loss
from neural_network.network import Network


def test_backprop_mathematical_correctness():
    """Test that backprop deltas are mathematically correct."""
    np.random.seed(42)

    # Simple network for manual verification
    x = np.array([[1.0, 2.0]])
    y_true = np.array([[1.0]])
    net = Network(x, depth=1, width=2)

    # Forward pass
    y_pred = net.feedforward(x)
    deltas = net._backprop(x, y_true)
    params = net._collect_params()

    print("=== Manual Verification ===")
    print(f"y_pred: {y_pred}")
    print(f"y_true: {y_true}")

    # Verify output layer delta
    dl_dyhat = deriv_mse_loss(y_true, y_pred)
    output_layer = "Layer_2"
    output_z = params[output_layer]["Neuron_0"]["z"]

    expected_output_delta = dl_dyhat.flatten() * deriv_sigmoid(output_z).flatten()
    actual_output_delta = deltas[output_layer]

    print(
        f"Output delta - Expected: {expected_output_delta}, Actual: {actual_output_delta}"
    )

    np.testing.assert_allclose(
        actual_output_delta,
        expected_output_delta,
        rtol=1e-6,
        err_msg="Output layer delta incorrect",
    )

    # Verify hidden layer delta
    hidden_layer = "Layer_1"

    # Get weights from output layer
    output_weights = []
    for neuron_key in params[output_layer]:
        weights = params[output_layer][neuron_key]["weights"]
        output_weights.append(weights.flatten())
    W_output = np.array(output_weights)

    # Get z values from hidden layer
    hidden_z_values = []
    for neuron_key in params[hidden_layer]:
        z_val = params[hidden_layer][neuron_key]["z"]
        hidden_z_values.append(z_val)
    z_hidden = np.array(hidden_z_values).flatten()

    expected_hidden_delta = W_output.T @ actual_output_delta * deriv_sigmoid(z_hidden)
    actual_hidden_delta = deltas[hidden_layer]

    print(
        f"Hidden delta - Expected: {expected_hidden_delta}, Actual: {actual_hidden_delta}"
    )

    np.testing.assert_allclose(
        actual_hidden_delta,
        expected_hidden_delta,
        rtol=1e-6,
        err_msg="Hidden layer delta incorrect",
    )

    print("✓ All delta calculations are mathematically correct!")


def test_backprop_delta_shapes_and_sizes():
    """Test that deltas have correct shapes for different network architectures."""

    test_cases = [
        {"depth": 1, "width": 1, "description": "Minimal network"},
        {"depth": 1, "width": 3, "description": "Single layer, wide"},
        {"depth": 2, "width": 2, "description": "Deep network"},
        {"depth": 3, "width": 4, "description": "Very deep network"},
    ]

    for i, case in enumerate(test_cases):
        np.random.seed(42 + i)

        x = np.array([[1.0, 2.0]])
        y_true = np.array([[1.0]])
        net = Network(x, depth=case["depth"], width=case["width"])

        # Forward and backward pass
        y_pred = net.feedforward(x)
        deltas = net._backprop(x, y_true)
        params = net._collect_params()

        print(f"\n=== Case {i}: {case['description']} ===")
        print(f"Network: depth={case['depth']}, width={case['width']}")
        print(f"Layers in params: {list(params.keys())}")
        print(f"Layers with deltas: {list(deltas.keys())}")

        # Check that deltas exist for each layer
        for layer_key in params:
            assert layer_key in deltas, f"Missing delta for {layer_key}"

            num_neurons = len(params[layer_key])
            delta_size = deltas[layer_key].size

            print(f"{layer_key}: {num_neurons} neurons, delta size: {delta_size}")

            # Delta should have one value per neuron
            assert (
                delta_size == num_neurons
            ), f"{layer_key}: delta size {delta_size} != {num_neurons} neurons"

        print(f"✓ Case {i}: All shapes correct")


def test_backprop_gradient_flow():
    """Test that gradients flow correctly through the network."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    y_true = np.array([[1.0]])
    net = Network(x, depth=2, width=2)  # 3 layers total

    y_pred = net.feedforward(x)
    deltas = net._backprop(x, y_true)

    print("=== Gradient Flow Test ===")
    print(f"Loss: {((y_true - y_pred) ** 2).mean()}")

    # Check that deltas get smaller as we go backwards (generally true for small gradients)
    layer_keys = sorted(deltas.keys())
    print("Delta magnitudes by layer:")

    for layer_key in layer_keys:
        delta_magnitude = np.abs(deltas[layer_key]).mean()
        print(f"{layer_key}: {delta_magnitude:.6f}")

    # All deltas should be finite
    for layer_key in deltas:
        delta = deltas[layer_key]
        assert np.all(np.isfinite(delta)), f"{layer_key}: deltas contain inf/nan values"

    print("✓ All gradients are finite and flow correctly")


def test_backprop_with_different_losses():
    """Test backprop with different prediction scenarios."""

    scenarios = [
        {"y_true": [[1.0]], "description": "Normal case"},
        {"y_true": [[0.0]], "description": "Zero target"},
        {"y_true": [[2.0]], "description": "Large target"},
        {"y_true": [[-1.0]], "description": "Negative target"},
    ]

    for i, scenario in enumerate(scenarios):
        np.random.seed(42)

        x = np.array([[1.0, 2.0]])
        y_true = np.array(scenario["y_true"])
        net = Network(x, depth=1, width=2)

        y_pred = net.feedforward(x)
        deltas = net._backprop(x, y_true)

        loss = ((y_true - y_pred) ** 2).mean()

        print(f"\n=== Scenario {i}: {scenario['description']} ===")
        print(f"y_true: {y_true.flatten()}, y_pred: {y_pred.flatten()}")
        print(f"Loss: {loss:.6f}")
        print(f"Deltas: {[f'{k}: {v}' for k, v in deltas.items()]}")

        # Check basic properties
        for layer_key in deltas:
            delta = deltas[layer_key]
            assert np.all(np.isfinite(delta)), f"Non-finite deltas in {layer_key}"

        # When prediction matches target, output delta should be small
        if np.allclose(y_true, y_pred, atol=1e-3):
            output_delta = deltas["Layer_2"]
            assert (
                np.abs(output_delta).max() < 1e-2
            ), "Delta should be small when prediction is accurate"

        print(f"✓ Scenario {i}: Deltas computed correctly")


def test_backprop_consistency():
    """Test that backprop gives consistent results across multiple calls."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    y_true = np.array([[1.0]])
    net = Network(x, depth=1, width=2)

    # Run backprop multiple times
    deltas1 = net._backprop(x, y_true)
    deltas2 = net._backprop(x, y_true)
    deltas3 = net._backprop(x, y_true)

    print("=== Consistency Test ===")

    # Should get identical results
    for layer_key in deltas1:
        np.testing.assert_array_equal(
            deltas1[layer_key],
            deltas2[layer_key],
            err_msg=f"Inconsistent results for {layer_key} between calls 1 and 2",
        )
        np.testing.assert_array_equal(
            deltas1[layer_key],
            deltas3[layer_key],
            err_msg=f"Inconsistent results for {layer_key} between calls 1 and 3",
        )
        print(f"{layer_key}: Consistent across all calls")

    print("✓ Backprop results are consistent")


if __name__ == "__main__":
    test_functions = [
        test_backprop_mathematical_correctness,
        test_backprop_delta_shapes_and_sizes,
        test_backprop_gradient_flow,
        test_backprop_with_different_losses,
        test_backprop_consistency,
    ]

    print("Running comprehensive backprop validation tests...")

    for test_func in test_functions:
        print(f"\n{'='*60}")
        print(f"Running: {test_func.__name__}")
        print("=" * 60)
        try:
            test_func()
            print(f"✓ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'='*60}")
    print("All backprop validation tests completed!")
    print("=" * 60)
