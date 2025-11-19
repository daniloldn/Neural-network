import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.activation import sigmoid
from neural_network.network import Network


def test_collect_params_basic_functionality():
    """Test that _collect_params runs without errors and returns a dict."""
    np.random.seed(42)  # For reproducible tests

    # Create simple network
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    net = Network(x, depth=1, width=2)

    # Should not raise any exceptions
    params = net._collect_params()

    # Should return a dictionary
    assert isinstance(params, dict), f"Expected dict, got {type(params)}"


def test_collect_params_structure_single_layer():
    """Test parameter collection structure for single layer network."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)  # Only init and output layers

    params = net._collect_params()

    # Should have structure organized by layers
    expected_layers = ["Layer_1", "Layer_2"]  # init and output layers

    for layer_key in expected_layers:
        assert layer_key in params, f"Missing '{layer_key}' in params dict"
        assert isinstance(params[layer_key], dict), f"{layer_key} should be a dict"

    # Each layer should have neurons with weights and bias
    for layer_key in expected_layers:
        layer_params = params[layer_key]
        assert len(layer_params) > 0, f"{layer_key} should have neurons"

        for neuron_key in layer_params:
            neuron_params = layer_params[neuron_key]
            assert "weights" in neuron_params, f"{neuron_key} missing weights"
            assert "bias" in neuron_params, f"{neuron_key} missing bias"
            assert "z" in neuron_params, f"{neuron_key} missing z"


def test_collect_params_layer_ids():
    """Test that all layers are included with correct IDs."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0, 3.0]])
    net = Network(x, depth=2, width=3)  # init, 1 hidden, output = 3 layers

    params = net._collect_params()

    # Should have 3 layers: Layer_1 (init), Layer_2 (hidden), Layer_3 (output)
    expected_layers = ["Layer_1", "Layer_2", "Layer_3"]

    for layer_key in expected_layers:
        assert layer_key in params, f"Missing {layer_key}"
        assert isinstance(params[layer_key], dict), f"{layer_key} should be dict"
        assert len(params[layer_key]) > 0, f"{layer_key} should have neurons"

    # Verify layer structure
    print(f"Found layers: {list(params.keys())}")
    assert len(params) == 3, f"Expected 3 layers, got {len(params)}"


def test_collect_params_neuron_weights_and_biases():
    """Test that weights and biases are collected for each neuron."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    params = net._collect_params()

    # Should have init layer (Layer_1) and output layer (Layer_2)
    assert "Layer_1" in params, "Missing init layer"
    assert "Layer_2" in params, "Missing output layer"

    # Check init layer parameters
    init_params = params["Layer_1"]
    assert (
        len(init_params) == 2
    ), f"Init layer should have 2 neurons, got {len(init_params)}"

    for neuron_key in init_params:
        neuron_data = init_params[neuron_key]
        assert "weights" in neuron_data, f"{neuron_key} missing weights"
        assert "bias" in neuron_data, f"{neuron_key} missing bias"

        # Check shapes - init layer takes 2 inputs
        weights = neuron_data["weights"]
        bias = neuron_data["bias"]
        assert weights.shape == (2, 1), f"Init weights shape: {weights.shape}"
        assert isinstance(bias, (int, float, np.number)), f"Bias type: {type(bias)}"

    # Check output layer parameters
    output_params = params["Layer_2"]
    assert (
        len(output_params) == 1
    ), f"Output layer should have 1 neuron, got {len(output_params)}"

    for neuron_key in output_params:
        neuron_data = output_params[neuron_key]
        assert "weights" in neuron_data, f"{neuron_key} missing weights"
        assert "bias" in neuron_data, f"{neuron_key} missing bias"

        # Check shapes - output layer takes 2 inputs (from 2 init neurons)
        weights = neuron_data["weights"]
        bias = neuron_data["bias"]
        assert weights.shape == (2, 1), f"Output weights shape: {weights.shape}"
        assert isinstance(bias, (int, float, np.number)), f"Bias type: {type(bias)}"


def test_collect_params_consistency_across_calls():
    """Test that multiple calls to _collect_params return consistent results."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    params1 = net._collect_params()
    params2 = net._collect_params()

    # Should return the same structure
    assert type(params1) == type(params2), "Return types should be consistent"
    assert set(params1.keys()) == set(params2.keys()), "Should have same layer keys"

    # Check that weights and biases are the same (same references)
    for layer_key in params1:
        assert layer_key in params2, f"Layer {layer_key} missing in second call"
        layer1 = params1[layer_key]
        layer2 = params2[layer_key]

        assert set(layer1.keys()) == set(
            layer2.keys()
        ), f"Neuron keys differ in {layer_key}"

        for neuron_key in layer1:
            neuron1 = layer1[neuron_key]
            neuron2 = layer2[neuron_key]

            # Same arrays should have same values
            np.testing.assert_array_equal(
                neuron1["weights"],
                neuron2["weights"],
                err_msg=f"Weights differ for {layer_key}.{neuron_key}",
            )
            assert (
                neuron1["bias"] == neuron2["bias"]
            ), f"Bias differs for {layer_key}.{neuron_key}"

    # The function should not modify the network
    assert len(net.init_layer.neurons) > 0, "Network neurons should still exist"
    assert len(net.output_layer.neurons) > 0, "Network neurons should still exist"


def test_collect_params_with_different_network_sizes():
    """Test parameter collection with different network architectures."""
    np.random.seed(42)

    test_cases = [
        {"inputs": 2, "depth": 1, "width": 1, "expected_layers": 2},  # Minimal network
        {"inputs": 3, "depth": 1, "width": 3, "expected_layers": 2},  # Single layer
        {
            "inputs": 2,
            "depth": 3,
            "width": 2,
            "expected_layers": 4,
        },  # Deep network (init + 2 hidden + output)
        {
            "inputs": 4,
            "depth": 2,
            "width": 5,
            "expected_layers": 3,
        },  # Wide network (init + 1 hidden + output)
    ]

    for i, case in enumerate(test_cases):
        x = np.random.randn(2, case["inputs"])  # 2 samples
        net = Network(x, depth=case["depth"], width=case["width"])

        params = net._collect_params()
        assert params is not None, f"Case {i}: Params should not be None"

        # Check expected number of layers
        actual_layers = len(params)
        expected_layers = case["expected_layers"]
        assert (
            actual_layers == expected_layers
        ), f"Case {i}: Expected {expected_layers} layers, got {actual_layers}"

        # Count total neurons across all layers
        total_neurons = sum(len(layer_params) for layer_params in params.values())

        print(
            f"Case {i}: {case} -> Layers: {actual_layers}, Total neurons: {total_neurons}"
        )
        assert total_neurons > 0, f"Case {i}: Should have neurons in network"


def test_collect_params_parameter_shapes():
    """Test that collected parameters have expected shapes."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0, 3.0]])  # 3 inputs
    net = Network(x, depth=1, width=2)

    params = net._collect_params()

    # Check init layer (Layer_1) - should have 2 neurons with 3 inputs each
    init_params = params["Layer_1"]
    assert (
        len(init_params) == 2
    ), f"Init layer should have 2 neurons, got {len(init_params)}"

    for neuron_key in init_params:
        neuron_data = init_params[neuron_key]
        weights = neuron_data["weights"]
        bias = neuron_data["bias"]

        assert weights.shape == (3, 1), f"Init neuron weights shape: {weights.shape}"
        assert isinstance(
            bias, (int, float, np.number)
        ), f"Bias should be scalar: {type(bias)}"

    # Check output layer (Layer_2) - should have 1 neuron with 2 inputs
    output_params = params["Layer_2"]
    assert (
        len(output_params) == 1
    ), f"Output layer should have 1 neuron, got {len(output_params)}"

    for neuron_key in output_params:
        neuron_data = output_params[neuron_key]
        weights = neuron_data["weights"]
        bias = neuron_data["bias"]

        assert weights.shape == (2, 1), f"Output neuron weights shape: {weights.shape}"
        assert isinstance(
            bias, (int, float, np.number)
        ), f"Bias should be scalar: {type(bias)}"


def test_collect_params_function_behavior_analysis():
    """Analyze the actual behavior of _collect_params to understand structure."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=2, width=2)  # init + 1 hidden + output

    print("=== Network Structure Analysis ===")
    print(f"Init layer ID: {net.init_layer.id}")
    print(f"Hidden layers count: {len(net.hidden_layer)}")
    if net.hidden_layer:
        print(f"Hidden layer IDs: {[layer.id for layer in net.hidden_layer]}")
    print(f"Output layer ID: {net.output_layer.id}")

    # Test the function
    params = net._collect_params()
    print(f"\n=== Function Output Structure ===")
    print(f"Number of layers collected: {len(params)}")
    print(f"Layer keys: {list(params.keys())}")

    for layer_key, layer_data in params.items():
        print(f"\n{layer_key}:")
        print(f"  Neurons: {list(layer_data.keys())}")
        for neuron_key, neuron_data in layer_data.items():
            weights_shape = neuron_data["weights"].shape
            print(
                f"    {neuron_key}: weights {weights_shape}, bias {type(neuron_data['bias'])}"
            )

    # Validate structure
    assert isinstance(params, dict), "Should return a dict"
    assert len(params) > 0, "Should have layers"

    for layer_key, layer_data in params.items():
        assert isinstance(layer_data, dict), f"{layer_key} should be a dict"
        assert len(layer_data) > 0, f"{layer_key} should have neurons"

        for neuron_key, neuron_data in layer_data.items():
            assert "weights" in neuron_data, f"{neuron_key} missing weights"
            assert "bias" in neuron_data, f"{neuron_key} missing bias"


def test_collect_params_deep_network():
    """Test parameter collection for a deeper network with multiple hidden layers."""
    np.random.seed(42)

    x = np.array([[1.0, 2.0, 3.0]])  # 3 inputs
    net = Network(x, depth=3, width=2)  # init + 2 hidden + output = 4 layers

    params = net._collect_params()

    # Should have 4 layers: Layer_1 (init), Layer_2, Layer_3 (hidden), Layer_4 (output)
    expected_layers = ["Layer_1", "Layer_2", "Layer_3", "Layer_4"]
    assert len(params) == 4, f"Expected 4 layers, got {len(params)}"

    for layer_key in expected_layers:
        assert layer_key in params, f"Missing {layer_key}"
        layer_data = params[layer_key]
        assert isinstance(layer_data, dict), f"{layer_key} should be dict"
        assert len(layer_data) > 0, f"{layer_key} should have neurons"

        # Each layer (except output) should have 'width' neurons
        if layer_key != "Layer_4":  # Not output layer
            assert (
                len(layer_data) == 2
            ), f"{layer_key} should have 2 neurons, got {len(layer_data)}"
        else:  # Output layer
            assert (
                len(layer_data) == 1
            ), f"Output layer should have 1 neuron, got {len(layer_data)}"


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_collect_params_basic_functionality,
        test_collect_params_structure_single_layer,
        test_collect_params_layer_ids,
        test_collect_params_neuron_weights_and_biases,
        test_collect_params_consistency_across_calls,
        test_collect_params_with_different_network_sizes,
        test_collect_params_parameter_shapes,
        test_collect_params_function_behavior_analysis,
        test_collect_params_deep_network,
    ]

    print("Running _collect_params tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("All tests completed!")
