#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.network import Network


def test_get_z_function():
    """Test the _get_z function works correctly"""
    print("=" * 60)
    print("Testing _get_z function")
    print("=" * 60)

    # Set seed for reproducible results
    np.random.seed(42)

    # Create a simple network
    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    # Do a forward pass to populate z values
    y_pred = net.feedforward(x)
    print(f"Network prediction: {y_pred}")

    # Collect parameters including z values
    params = net._collect_params()
    print(f"Collected layers: {list(params.keys())}")

    # Test _get_z for each layer
    for layer_key in params.keys():
        print(f"\nTesting _get_z for {layer_key}")

        # Check if neurons have z values
        layer = params[layer_key]
        print(f"  Neurons in {layer_key}: {list(layer.keys())}")

        for neuron_key, neuron_data in layer.items():
            z_val = neuron_data.get("z")
            print(f"    {neuron_key} z value: {z_val} (type: {type(z_val)})")

        # Test _get_z function
        try:
            z_array = net._get_z(params, layer_key)
            print(f"  _get_z result: {z_array}")
            print(f"  Shape: {z_array.shape}, Type: {type(z_array)}")
            print(f"  All finite: {np.all(np.isfinite(z_array))}")
        except Exception as e:
            print(f"  ERROR in _get_z: {e}")

    print("\n✓ _get_z function test completed")


def test_get_z_in_backprop():
    """Test how _get_z is used in backprop"""
    print("\n" + "=" * 60)
    print("Testing _get_z usage in backprop")
    print("=" * 60)

    # Set seed for reproducible results
    np.random.seed(42)

    # Create network
    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    # Forward pass
    y_pred = net.feedforward(x)
    y_true = np.array([[0.5]])

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")

    # Collect parameters
    params = net._collect_params()

    # Test backprop manually step by step
    print(f"\nTesting backprop step by step...")

    # Import loss function
    from neural_network.activation import deriv_sigmoid
    from neural_network.loss import deriv_mse_loss

    # Calculate loss gradient
    dl_dyhat = deriv_mse_loss(y_true, y_pred)
    print(f"Loss gradient: {dl_dyhat} (shape: {dl_dyhat.shape})")

    # Test output layer delta calculation
    output_layer_id = net.depth + 1
    output_layer_key = f"Layer_{output_layer_id}"
    print(f"Output layer key: {output_layer_key}")

    if output_layer_key in params:
        print(f"Output layer found in params")

        # Test _get_z for output layer
        try:
            z_array = net._get_z(params, output_layer_key)
            print(f"Output z values: {z_array}")

            # Calculate activation derivative
            activation_deriv = deriv_sigmoid(z_array)
            print(f"Activation derivative: {activation_deriv}")

            # Calculate delta
            delta_output = dl_dyhat.flatten() * activation_deriv
            print(f"Output delta: {delta_output}")
            print(f"Delta shape: {delta_output.shape}")

        except Exception as e:
            print(f"ERROR in output layer calculation: {e}")

    # Test hidden layers if they exist
    for layer_id in range(net.depth, 0, -1):
        layer_key = f"Layer_{layer_id}"
        next_layer_key = f"Layer_{layer_id + 1}"

        print(f"\nTesting hidden layer {layer_key}")

        if layer_key in params:
            try:
                z_array = net._get_z(params, layer_key)
                print(f"  {layer_key} z values: {z_array}")
                print(f"  Shape: {z_array.shape}")
            except Exception as e:
                print(f"  ERROR getting z for {layer_key}: {e}")

    # Test full backprop function
    print(f"\nTesting full _backprop function:")
    try:
        deltas = net._backprop(params, y_true, y_pred)
        print(f"Backprop successful!")
        print(f"Delta keys: {list(deltas.keys())}")
        for key, delta in deltas.items():
            print(f"  {key}: {delta} (shape: {delta.shape})")
    except Exception as e:
        print(f"ERROR in full backprop: {e}")
        import traceback

        traceback.print_exc()

    print("\n✓ Backprop test completed")


def test_network_structure():
    """Test network structure and parameter collection"""
    print("\n" + "=" * 60)
    print("Testing network structure")
    print("=" * 60)

    # Test different network sizes
    test_configs = [
        {"depth": 1, "width": 1, "input_size": 2},
        {"depth": 1, "width": 3, "input_size": 2},
        {"depth": 2, "width": 2, "input_size": 3},
    ]

    for i, config in enumerate(test_configs):
        print(
            f"\nTest {i+1}: depth={config['depth']}, width={config['width']}, input_size={config['input_size']}"
        )

        np.random.seed(42)
        x = np.random.rand(1, config["input_size"])

        try:
            net = Network(x, depth=config["depth"], width=config["width"])
            y_pred = net.feedforward(x)
            params = net._collect_params()

            print(f"  Network created successfully")
            print(f"  Prediction shape: {y_pred.shape}")
            print(f"  Parameter layers: {list(params.keys())}")

            # Test _get_z for all layers
            all_z_work = True
            for layer_key in params.keys():
                try:
                    z_array = net._get_z(params, layer_key)
                    print(f"    {layer_key} z shape: {z_array.shape}")
                except Exception as e:
                    print(f"    ERROR in {layer_key}: {e}")
                    all_z_work = False

            if all_z_work:
                print(f"  ✓ All _get_z calls successful")
            else:
                print(f"  ✗ Some _get_z calls failed")

        except Exception as e:
            print(f"  ERROR creating network: {e}")

    print("\n✓ Network structure test completed")


if __name__ == "__main__":
    print("Testing _get_z function and its implementation in backprop")
    print("=" * 70)

    test_get_z_function()
    test_get_z_in_backprop()
    test_network_structure()

    print("\n" + "=" * 70)
    print("All tests completed!")
