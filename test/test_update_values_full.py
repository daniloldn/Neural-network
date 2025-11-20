#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.network import Network


def test_update_values_comprehensive():
    """Comprehensive test of update_values function"""
    print("=" * 80)
    print("COMPREHENSIVE UPDATE_VALUES TESTING")
    print("=" * 80)

    # Set seed for reproducible results
    np.random.seed(42)

    # Create test data
    x = np.array([[1.0, 2.0]])
    y_true = np.array([[0.5]])
    learning_rate = 0.01

    print(f"Test configuration:")
    print(f"  Input: {x}")
    print(f"  Target: {y_true}")
    print(f"  Learning rate: {learning_rate}")

    # Initialize network
    net = Network(x, depth=1, width=2)

    # Forward pass and setup
    y_pred = net.feedforward(x)
    params = net._collect_params()
    deltas = net._backprop(params, y_true, y_pred)

    print(f"\nSetup results:")
    print(f"  Prediction: {y_pred}")
    print(f"  Layers: {list(params.keys())}")
    print(f"  Deltas: {list(deltas.keys())}")

    # Store original parameters for comparison
    original_params = {}
    for layer_key, layer in params.items():
        original_params[layer_key] = {}
        for neuron_key, neuron in layer.items():
            original_params[layer_key][neuron_key] = {
                "weights": neuron["weights"].copy(),
                "bias": neuron["bias"],
            }

    print(f"\nOriginal parameters:")
    for layer_key, layer in original_params.items():
        print(f"  {layer_key}:")
        for neuron_key, neuron in layer.items():
            print(
                f"    {neuron_key}: weights={neuron['weights'].flatten()}, bias={neuron['bias']}"
            )

    print(f"\nDeltas for update:")
    for layer_key, delta in deltas.items():
        print(f"  {layer_key}: {delta} (shape: {delta.shape}, type: {type(delta)})")

    # Test update_values
    print(f"\n{'-'*60}")
    print("TESTING UPDATE_VALUES")
    print(f"{'-'*60}")

    try:
        updated_params = net.update_values(x, params, deltas, learning_rate)
        print("✓ update_values completed successfully!")

        # Analyze the results
        print(f"\n{'-'*40}")
        print("PARAMETER CHANGE ANALYSIS")
        print(f"{'-'*40}")

        total_changes = 0
        for layer_key in params.keys():
            if layer_key in updated_params and layer_key in original_params:
                print(f"\n{layer_key}:")
                layer_changed = False

                for neuron_key in params[layer_key].keys():
                    if (
                        neuron_key in updated_params[layer_key]
                        and neuron_key in original_params[layer_key]
                    ):
                        # Weight changes
                        old_weights = original_params[layer_key][neuron_key]["weights"]
                        new_weights = updated_params[layer_key][neuron_key]["weights"]
                        weight_diff = np.abs(old_weights - new_weights)
                        weight_changed = np.any(weight_diff > 1e-10)

                        # Bias changes
                        old_bias = original_params[layer_key][neuron_key]["bias"]
                        new_bias = updated_params[layer_key][neuron_key]["bias"]
                        bias_diff = abs(old_bias - new_bias)
                        bias_changed = bias_diff > 1e-10

                        print(f"  {neuron_key}:")
                        print(f"    Weights - Old: {old_weights.flatten()}")
                        print(f"    Weights - New: {new_weights.flatten()}")
                        print(f"    Weight diff: {weight_diff.flatten()}")
                        print(f"    Weight changed: {weight_changed}")
                        print(f"    Bias - Old: {old_bias}")
                        print(f"    Bias - New: {new_bias}")
                        print(f"    Bias diff: {bias_diff}")
                        print(f"    Bias changed: {bias_changed}")

                        if weight_changed or bias_changed:
                            layer_changed = True
                            total_changes += 1

                print(f"  Layer {layer_key} had changes: {layer_changed}")
            else:
                print(f"\n{layer_key}: NOT FOUND IN RESULTS!")

        print(f"\nSummary:")
        print(f"  Total neurons with changes: {total_changes}")
        print(f"  Expected changes: Should be > 0 if gradients applied correctly")

        # Test if function returns updated parameters
        print(f"\n{'-'*40}")
        print("RETURN VALUE ANALYSIS")
        print(f"{'-'*40}")

        print(f"  Returned object type: {type(updated_params)}")
        print(f"  Returned keys: {list(updated_params.keys())}")
        print(f"  Same object as input: {updated_params is params}")

    except Exception as e:
        print(f"✗ update_values failed: {e}")
        print(f"\n{'-'*40}")
        print("ERROR ANALYSIS")
        print(f"{'-'*40}")

        import traceback

        traceback.print_exc()

        # Debug the inputs
        print(f"\nInput debugging:")
        print(f"  x type: {type(x)}, shape: {x.shape}")
        print(f"  params type: {type(params)}")
        print(f"  deltas type: {type(deltas)}")
        print(f"  learning_rate type: {type(learning_rate)}")

        # Check each delta
        for layer_key, delta in deltas.items():
            print(f"  {layer_key} delta:")
            print(f"    Type: {type(delta)}")
            print(f"    Shape: {delta.shape}")
            print(f"    Values: {delta}")
            for i, val in enumerate(delta):
                print(
                    f"      delta[{i}]: {val} (type: {type(val)}, shape: {val.shape if hasattr(val, 'shape') else 'no shape'})"
                )


def test_different_network_configurations():
    """Test update_values with different network sizes"""
    print(f"\n{'='*80}")
    print("TESTING DIFFERENT NETWORK CONFIGURATIONS")
    print(f"{'='*80}")

    configs = [
        {"name": "Minimal 1x1", "depth": 1, "width": 1, "input_size": 1},
        {"name": "Simple 1x2", "depth": 1, "width": 2, "input_size": 2},
        {"name": "Wide 1x3", "depth": 1, "width": 3, "input_size": 2},
        {"name": "Deep 2x2", "depth": 2, "width": 2, "input_size": 2},
    ]

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

            # Setup
            y_pred = net.feedforward(x)
            y_true = np.random.rand(*y_pred.shape)
            params = net._collect_params()
            deltas = net._backprop(params, y_true, y_pred)

            print(f"  Setup successful")
            print(f"    Input shape: {x.shape}")
            print(f"    Prediction shape: {y_pred.shape}")
            print(f"    Layers: {list(params.keys())}")
            print(f"    Deltas: {list(deltas.keys())}")

            # Test update_values
            updated_params = net.update_values(x, params, deltas, 0.01)

            print(f"  ✓ update_values successful")
            print(f"    Returned layers: {list(updated_params.keys())}")

            # Check if any parameters actually changed
            changes_detected = False
            for layer_key in params.keys():
                if layer_key in deltas:  # Only check layers that had deltas
                    layer = params[layer_key]
                    for neuron_key, neuron in layer.items():
                        # Check if this neuron's parameters are in updated_params
                        if (
                            layer_key in updated_params
                            and neuron_key in updated_params[layer_key]
                        ):

                            old_weights = neuron["weights"]
                            new_weights = updated_params[layer_key][neuron_key][
                                "weights"
                            ]
                            if np.any(np.abs(old_weights - new_weights) > 1e-10):
                                changes_detected = True
                                break
                    if changes_detected:
                        break

            print(f"    Parameter changes detected: {changes_detected}")

        except Exception as e:
            print(f"  ✗ {config['name']} failed: {e}")


def test_edge_cases():
    """Test edge cases and potential issues"""
    print(f"\n{'='*80}")
    print("TESTING EDGE CASES")
    print(f"{'='*80}")

    # Test 1: Very small learning rate
    print(f"\n{'-'*40}")
    print("Test 1: Very small learning rate")
    print(f"{'-'*40}")

    try:
        np.random.seed(42)
        x = np.array([[1.0, 2.0]])
        net = Network(x, depth=1, width=2)

        y_pred = net.feedforward(x)
        y_true = np.array([[0.5]])
        params = net._collect_params()
        deltas = net._backprop(params, y_true, y_pred)

        # Very small learning rate
        tiny_lr = 1e-8
        updated_params = net.update_values(x, params, deltas, tiny_lr)
        print(f"  ✓ Tiny learning rate ({tiny_lr}) handled successfully")

    except Exception as e:
        print(f"  ✗ Tiny learning rate failed: {e}")

    # Test 2: Large learning rate
    print(f"\n{'-'*40}")
    print("Test 2: Large learning rate")
    print(f"{'-'*40}")

    try:
        np.random.seed(42)
        x = np.array([[1.0, 2.0]])
        net = Network(x, depth=1, width=2)

        y_pred = net.feedforward(x)
        y_true = np.array([[0.5]])
        params = net._collect_params()
        deltas = net._backprop(params, y_true, y_pred)

        # Large learning rate
        large_lr = 10.0
        updated_params = net.update_values(x, params, deltas, large_lr)
        print(f"  ✓ Large learning rate ({large_lr}) handled successfully")

    except Exception as e:
        print(f"  ✗ Large learning rate failed: {e}")

    # Test 3: Zero deltas
    print(f"\n{'-'*40}")
    print("Test 3: Zero deltas")
    print(f"{'-'*40}")

    try:
        np.random.seed(42)
        x = np.array([[1.0, 2.0]])
        net = Network(x, depth=1, width=2)

        params = net._collect_params()

        # Create zero deltas manually
        zero_deltas = {}
        for layer_key, layer in params.items():
            num_neurons = len(layer)
            zero_deltas[layer_key] = np.zeros(num_neurons)

        updated_params = net.update_values(x, params, zero_deltas, 0.01)
        print(f"  ✓ Zero deltas handled successfully")

    except Exception as e:
        print(f"  ✗ Zero deltas failed: {e}")


if __name__ == "__main__":
    test_update_values_comprehensive()
    test_different_network_configurations()
    test_edge_cases()

    print(f"\n{'='*80}")
    print("ALL UPDATE_VALUES TESTS COMPLETED!")
    print(f"{'='*80}")
