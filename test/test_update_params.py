#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.network import Network


def test_update_params_basic():
    """Test basic functionality of _update_params"""
    print("=" * 70)
    print("Testing _update_params Basic Functionality")
    print("=" * 70)

    # Set seed for reproducible results
    np.random.seed(42)

    # Create a simple network
    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    print(f"Created network with depth={net.depth}, width=2")
    print(f"Network structure: {[type(layer).__name__ for layer in net.network]}")

    # Store original parameters for comparison
    print(f"\nOriginal network parameters:")
    original_weights = {}
    original_biases = {}

    for i, layer in enumerate(net.network):
        if isinstance(layer, list):
            print(f"Layer {i} (hidden list):")
            for j, hidden_layer in enumerate(layer):
                layer_id = hidden_layer.id
                print(f"  Hidden layer {j} (id: {layer_id}):")
                original_weights[layer_id] = {}
                original_biases[layer_id] = {}
                for neuron in hidden_layer.neurons:
                    neuron_id = neuron.id
                    weights = (
                        neuron.weights
                        if hasattr(neuron, "weights")
                        else getattr(neuron, "weight", None)
                    )
                    bias = neuron.bias
                    print(
                        f"    {neuron_id}: weights={weights.flatten() if hasattr(weights, 'flatten') else weights}, bias={bias}"
                    )
                    original_weights[layer_id][neuron_id] = (
                        weights.copy() if hasattr(weights, "copy") else weights
                    )
                    original_biases[layer_id][neuron_id] = bias
        else:
            layer_id = layer.id
            print(f"Layer {i} (id: {layer_id}):")
            original_weights[layer_id] = {}
            original_biases[layer_id] = {}
            for neuron in layer.neurons:
                neuron_id = neuron.id
                weights = (
                    neuron.weights
                    if hasattr(neuron, "weights")
                    else getattr(neuron, "weight", None)
                )
                bias = neuron.bias
                print(
                    f"  {neuron_id}: weights={weights.flatten() if hasattr(weights, 'flatten') else weights}, bias={bias}"
                )
                original_weights[layer_id][neuron_id] = (
                    weights.copy() if hasattr(weights, "copy") else weights
                )
                original_biases[layer_id][neuron_id] = bias

    # Create modified parameters dictionary
    print(f"\nCreating modified parameters...")
    params = net._collect_params()
    print(f"Collected params keys: {list(params.keys())}")

    # Show the structure of collected params
    for layer_key, layer_data in params.items():
        print(f"{layer_key}: {list(layer_data.keys())}")
        for neuron_key, neuron_data in layer_data.items():
            print(f"  {neuron_key}: {list(neuron_data.keys())}")

    # Modify the parameters (add small changes)
    modified_params = {}
    for layer_key, layer_data in params.items():
        modified_params[layer_key] = {}
        for neuron_key, neuron_data in layer_data.items():
            modified_params[layer_key][neuron_key] = {
                "weights": neuron_data["weights"] + 0.1,  # Add 0.1 to all weights
                "bias": neuron_data["bias"] + 0.05,  # Add 0.05 to bias
                "z": neuron_data["z"],  # Keep z unchanged
            }

    print(f"\nModified parameters (added 0.1 to weights, 0.05 to bias)")

    # Test _update_params
    print(f"\nTesting _update_params...")
    try:
        result = net._update_params(modified_params)
        print(f"✓ _update_params completed, returned: {result}")

        # Verify the parameters were actually updated in the network
        print(f"\nVerifying parameter updates...")
        verification_passed = True

        for i, layer in enumerate(net.network):
            if isinstance(layer, list):
                for j, hidden_layer in enumerate(layer):
                    layer_id = hidden_layer.id
                    if layer_id in modified_params:
                        for neuron in hidden_layer.neurons:
                            neuron_id = neuron.id
                            if neuron_id in modified_params[layer_id]:
                                # Check weights
                                current_weights = (
                                    neuron.weights
                                    if hasattr(neuron, "weights")
                                    else getattr(neuron, "weight", None)
                                )
                                expected_weights = modified_params[layer_id][neuron_id][
                                    "weights"
                                ]

                                if not np.allclose(
                                    current_weights, expected_weights, atol=1e-10
                                ):
                                    print(
                                        f"✗ {layer_id}.{neuron_id} weights not updated correctly"
                                    )
                                    print(f"  Expected: {expected_weights.flatten()}")
                                    print(
                                        f"  Actual: {current_weights.flatten() if hasattr(current_weights, 'flatten') else current_weights}"
                                    )
                                    verification_passed = False
                                else:
                                    print(
                                        f"✓ {layer_id}.{neuron_id} weights updated correctly"
                                    )

                                # Check bias
                                current_bias = neuron.bias
                                expected_bias = modified_params[layer_id][neuron_id][
                                    "bias"
                                ]

                                if abs(current_bias - expected_bias) > 1e-10:
                                    print(
                                        f"✗ {layer_id}.{neuron_id} bias not updated correctly"
                                    )
                                    print(f"  Expected: {expected_bias}")
                                    print(f"  Actual: {current_bias}")
                                    verification_passed = False
                                else:
                                    print(
                                        f"✓ {layer_id}.{neuron_id} bias updated correctly"
                                    )
            else:
                layer_id = layer.id
                if layer_id in modified_params:
                    for neuron in layer.neurons:
                        neuron_id = neuron.id
                        if neuron_id in modified_params[layer_id]:
                            # Check weights
                            current_weights = (
                                neuron.weights
                                if hasattr(neuron, "weights")
                                else getattr(neuron, "weight", None)
                            )
                            expected_weights = modified_params[layer_id][neuron_id][
                                "weights"
                            ]

                            if not np.allclose(
                                current_weights, expected_weights, atol=1e-10
                            ):
                                print(
                                    f"✗ {layer_id}.{neuron_id} weights not updated correctly"
                                )
                                print(f"  Expected: {expected_weights.flatten()}")
                                print(
                                    f"  Actual: {current_weights.flatten() if hasattr(current_weights, 'flatten') else current_weights}"
                                )
                                verification_passed = False
                            else:
                                print(
                                    f"✓ {layer_id}.{neuron_id} weights updated correctly"
                                )

                            # Check bias
                            current_bias = neuron.bias
                            expected_bias = modified_params[layer_id][neuron_id]["bias"]

                            if abs(current_bias - expected_bias) > 1e-10:
                                print(
                                    f"✗ {layer_id}.{neuron_id} bias not updated correctly"
                                )
                                print(f"  Expected: {expected_bias}")
                                print(f"  Actual: {current_bias}")
                                verification_passed = False
                            else:
                                print(
                                    f"✓ {layer_id}.{neuron_id} bias updated correctly"
                                )

        if verification_passed:
            print(f"\n🎉 All parameter updates verified successfully!")
        else:
            print(f"\n⚠️ Some parameter updates failed verification")

        return verification_passed

    except Exception as e:
        print(f"✗ _update_params failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_update_params_different_networks():
    """Test _update_params with different network configurations"""
    print(f"\n{'='*70}")
    print("Testing _update_params with Different Network Configurations")
    print(f"{'='*70}")

    configs = [
        {"name": "Simple 1x1", "depth": 1, "width": 1, "input_size": 2},
        {"name": "Wide 1x3", "depth": 1, "width": 3, "input_size": 2},
        {"name": "Deep 2x2", "depth": 2, "width": 2, "input_size": 3},
    ]

    success_count = 0

    for config in configs:
        print(f"\n{'-'*50}")
        print(f"Testing: {config['name']}")
        print(
            f"Depth: {config['depth']}, Width: {config['width']}, Input size: {config['input_size']}"
        )
        print(f"{'-'*50}")

        try:
            np.random.seed(42)
            x = np.random.rand(1, config["input_size"])
            net = Network(x, depth=config["depth"], width=config["width"])

            # Get current parameters
            original_params = net._collect_params()

            # Create modified parameters
            modified_params = {}
            for layer_key, layer_data in original_params.items():
                modified_params[layer_key] = {}
                for neuron_key, neuron_data in layer_data.items():
                    modified_params[layer_key][neuron_key] = {
                        "weights": neuron_data["weights"] * 1.1,  # Multiply by 1.1
                        "bias": neuron_data["bias"] + 0.2,  # Add 0.2
                        "z": neuron_data["z"],
                    }

            # Test update
            net._update_params(modified_params)

            # Quick verification - check if at least one parameter changed
            new_params = net._collect_params()
            changed = False

            for layer_key in original_params:
                if layer_key in new_params:
                    for neuron_key in original_params[layer_key]:
                        if neuron_key in new_params[layer_key]:
                            old_weights = original_params[layer_key][neuron_key][
                                "weights"
                            ]
                            new_weights = new_params[layer_key][neuron_key]["weights"]
                            if not np.allclose(old_weights, new_weights, atol=1e-10):
                                changed = True
                                break
                    if changed:
                        break

            if changed:
                print(f"  ✓ Parameters successfully updated")
                success_count += 1
            else:
                print(f"  ⚠️ No parameter changes detected")

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nSummary: {success_count}/{len(configs)} configurations successful")
    return success_count == len(configs)


def test_update_params_edge_cases():
    """Test edge cases for _update_params"""
    print(f"\n{'='*70}")
    print("Testing _update_params Edge Cases")
    print(f"{'='*70}")

    # Test 1: Empty parameters dict
    print(f"\nTest 1: Empty parameters dict")
    try:
        np.random.seed(42)
        x = np.array([[1.0, 2.0]])
        net = Network(x, depth=1, width=1)

        empty_params = {}
        net._update_params(empty_params)
        print(f"  ✓ Empty params handled gracefully")

    except Exception as e:
        print(f"  ✗ Empty params failed: {e}")

    # Test 2: Mismatched parameter structure
    print(f"\nTest 2: Mismatched parameter keys")
    try:
        np.random.seed(42)
        x = np.array([[1.0, 2.0]])
        net = Network(x, depth=1, width=1)

        wrong_params = {
            "NonExistentLayer": {
                "NonExistentNeuron": {
                    "weights": np.array([[1.0]]),
                    "bias": 0.5,
                    "z": np.array([[0.0]]),
                }
            }
        }

        net._update_params(wrong_params)
        print(f"  ✓ Mismatched keys handled gracefully")

    except Exception as e:
        print(f"  ✗ Mismatched keys failed: {e}")


if __name__ == "__main__":
    print("TESTING _update_params FUNCTION")
    print("=" * 70)

    test1 = test_update_params_basic()
    test2 = test_update_params_different_networks()
    test_update_params_edge_cases()

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Basic functionality: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Different networks: {'✓ PASS' if test2 else '✗ FAIL'}")

    if test1 and test2:
        print(f"\n🎉 _update_params function works correctly!")
    else:
        print(f"\n⚠️ _update_params has issues that need fixing")
