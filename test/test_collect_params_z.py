import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.network import Network


def test_collect_params_with_z_values():
    """Test that _collect_params correctly handles z values before and after forward pass."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    # Test before forward pass - z should be None
    params_before = net._collect_params()

    # Check that z values are None before forward pass
    for layer_key, layer_data in params_before.items():
        for neuron_key, neuron_data in layer_data.items():
            assert "z" in neuron_data, f"{neuron_key} missing z attribute"
            assert (
                neuron_data["z"] is None
            ), f"{neuron_key} z should be None before forward pass, got {neuron_data['z']}"

    # Perform forward pass
    output = net.feedforward(x)

    # Test after forward pass - z should have values
    params_after = net._collect_params()

    # Check that z values are populated after forward pass
    for layer_key, layer_data in params_after.items():
        for neuron_key, neuron_data in layer_data.items():
            assert "z" in neuron_data, f"{neuron_key} missing z attribute"
            z_value = neuron_data["z"]
            assert (
                z_value is not None
            ), f"{neuron_key} z should not be None after forward pass"

            # z should be a numpy array or scalar
            assert isinstance(
                z_value, (np.ndarray, np.number, int, float)
            ), f"{neuron_key} z should be numeric, got {type(z_value)}"

    print(
        "✓ z values correctly handled before (None) and after (populated) forward pass"
    )


def test_collect_params_z_structure():
    """Test the complete structure including z values after forward pass."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    # Perform forward pass to populate z values
    output = net.feedforward(x)

    params = net._collect_params()

    # Verify structure includes all expected keys
    expected_keys = {"weights", "bias", "z"}

    for layer_key, layer_data in params.items():
        for neuron_key, neuron_data in layer_data.items():
            actual_keys = set(neuron_data.keys())
            assert (
                actual_keys == expected_keys
            ), f"{neuron_key} keys {actual_keys} != expected {expected_keys}"

            # Verify types
            assert hasattr(
                neuron_data["weights"], "shape"
            ), "weights should be array-like"
            assert isinstance(
                neuron_data["bias"], (int, float, np.number)
            ), "bias should be scalar"
            assert (
                neuron_data["z"] is not None
            ), "z should be populated after forward pass"

    print(f"✓ All neurons have complete structure: {expected_keys}")


def test_collect_params_z_consistency():
    """Test that z values are consistent with manual calculation."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=1)  # Simple network for easy verification

    # Perform forward pass
    output = net.feedforward(x)

    params = net._collect_params()

    # Get init layer neuron (first neuron)
    init_neuron_params = params["Layer_1"]["Neuron_0"]
    weights = init_neuron_params["weights"]
    bias = init_neuron_params["bias"]
    z_collected = init_neuron_params["z"]

    # Manual calculation: z = x @ weights + bias
    z_manual = np.dot(x, weights) + bias

    # Should match (within numerical precision)
    np.testing.assert_allclose(
        z_collected,
        z_manual,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Collected z {z_collected} != manual z {z_manual}",
    )

    print(
        f"✓ z values match manual calculation: {z_collected.flatten()} ≈ {z_manual.flatten()}"
    )


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_collect_params_with_z_values,
        test_collect_params_z_structure,
        test_collect_params_z_consistency,
    ]

    print("Running _collect_params z-value tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("All z-value tests completed!")
