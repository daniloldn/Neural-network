#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.network import Network


def test_backprop_basic_functionality():
    """Test basic backprop functionality with correct signature"""
    print("=" * 60)
    print("Testing Backprop Basic Functionality")
    print("=" * 60)

    # Set seed for reproducible results
    np.random.seed(42)

    # Test 1: Simple network
    print("\nTest 1: Simple network (depth=1, width=2)")
    print("-" * 40)

    x = np.array([[1.0, 2.0]])
    net = Network(x, depth=1, width=2)

    # Forward pass
    y_pred = net.feedforward(x)
    y_true = np.array([[0.5]])
    params = net._collect_params()

    print(f"Input: {x}")
    print(f"Prediction: {y_pred}")
    print(f"Target: {y_true}")
    print(f"Network layers: {list(params.keys())}")

    # Test backprop with correct signature
    try:
        deltas = net._backprop(params, y_true, y_pred)
        print(f"✓ Backprop successful")
        print(f"Delta keys: {list(deltas.keys())}")

        for layer_key, delta in deltas.items():
            print(f"  {layer_key}: shape={delta.shape}, values={delta}")

        # Verify all layers have deltas
        expected_layers = set(params.keys())
        actual_layers = set(deltas.keys())

        if expected_layers == actual_layers:
            print(f"✓ All layers have deltas")
        else:
            print(f"✗ Missing deltas for: {expected_layers - actual_layers}")

    except Exception as e:
        print(f"✗ Backprop failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test 2: Deeper network
    print("\nTest 2: Deeper network (depth=2, width=2)")
    print("-" * 40)

    try:
        x = np.array([[1.0, 2.0]])
        net = Network(x, depth=2, width=2)

        y_pred = net.feedforward(x)
        y_true = np.array([[0.3]])
        params = net._collect_params()

        print(f"Network layers: {list(params.keys())}")

        deltas = net._backprop(params, y_true, y_pred)
        print(f"✓ Deep network backprop successful")
        print(f"Delta keys: {list(deltas.keys())}")

        for layer_key, delta in deltas.items():
            print(f"  {layer_key}: shape={delta.shape}, values={delta}")

    except Exception as e:
        print(f"✗ Deep network backprop failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_backprop_mathematical_consistency():
    """Test mathematical consistency of backprop"""
    print("\n" + "=" * 60)
    print("Testing Backprop Mathematical Consistency")
    print("=" * 60)

    np.random.seed(42)

    # Test with known values
    x = np.array([[1.0, 0.0]])  # Simple input
    net = Network(x, depth=1, width=1)  # Minimal network

    # Forward pass
    y_pred = net.feedforward(x)
    y_true = np.array([[0.0]])  # Target
    params = net._collect_params()

    print(f"Input: {x}")
    print(f"Prediction: {y_pred}")
    print(f"Target: {y_true}")
    print(f"Loss: {0.5 * (y_true - y_pred)**2}")

    # Manual calculation check
    from neural_network.activation import deriv_sigmoid
    from neural_network.loss import deriv_mse_loss

    manual_loss_grad = deriv_mse_loss(y_true, y_pred)
    print(f"Manual loss gradient: {manual_loss_grad}")

    # Get z values and compute manual delta
    z_output = net._get_z(params, "Layer_2")
    manual_activation_deriv = deriv_sigmoid(z_output)
    manual_delta = manual_loss_grad.flatten() * manual_activation_deriv

    print(f"Manual output delta: {manual_delta}")

    # Compare with backprop result
    deltas = net._backprop(params, y_true, y_pred)
    backprop_delta = deltas["Layer_2"]

    print(f"Backprop output delta: {backprop_delta}")
    print(f"Difference: {np.abs(manual_delta - backprop_delta)}")

    # Check if they're close
    if np.allclose(manual_delta, backprop_delta, atol=1e-10):
        print("✓ Manual calculation matches backprop")
        return True
    else:
        print("✗ Manual calculation differs from backprop")
        return False


def test_backprop_different_network_sizes():
    """Test backprop with various network configurations"""
    print("\n" + "=" * 60)
    print("Testing Backprop with Different Network Sizes")
    print("=" * 60)

    configs = [
        {"name": "Tiny 1x1", "depth": 1, "width": 1, "input_size": 1},
        {"name": "Small 1x2", "depth": 1, "width": 2, "input_size": 2},
        {"name": "Wide 1x5", "depth": 1, "width": 5, "input_size": 3},
        {"name": "Deep 3x2", "depth": 3, "width": 2, "input_size": 2},
    ]

    success_count = 0
    total_count = len(configs)

    for config in configs:
        print(
            f"\n{config['name']} (depth={config['depth']}, width={config['width']}, input={config['input_size']})"
        )
        print("-" * 30)

        try:
            np.random.seed(42)
            x = np.random.rand(1, config["input_size"])
            net = Network(x, depth=config["depth"], width=config["width"])

            y_pred = net.feedforward(x)
            y_true = np.random.rand(*y_pred.shape)
            params = net._collect_params()

            deltas = net._backprop(params, y_true, y_pred)

            print(f"  ✓ Success")
            print(f"  Layers: {len(params)} -> {len(deltas)} deltas")

            # Check delta shapes match layer neuron counts
            shape_ok = True
            for layer_key, layer in params.items():
                if layer_key in deltas:
                    expected_neurons = len(layer)
                    actual_delta_size = deltas[layer_key].size
                    if expected_neurons != actual_delta_size:
                        print(
                            f"  ✗ {layer_key}: expected {expected_neurons} deltas, got {actual_delta_size}"
                        )
                        shape_ok = False

            if shape_ok:
                print(f"  ✓ All delta shapes correct")
                success_count += 1

        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print(f"\nSummary: {success_count}/{total_count} configurations successful")
    return success_count == total_count


if __name__ == "__main__":
    print("BACKPROP FUNCTIONALITY TESTS")
    print("=" * 60)

    test1 = test_backprop_basic_functionality()
    test2 = test_backprop_mathematical_consistency()
    test3 = test_backprop_different_network_sizes()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Basic functionality: {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Mathematical consistency: {'✓ PASS' if test2 else '✗ FAIL'}")
    print(f"Different network sizes: {'✓ PASS' if test3 else '✗ FAIL'}")

    if all([test1, test2, test3]):
        print("\n🎉 ALL BACKPROP TESTS PASSED!")
    else:
        print("\n⚠️  Some backprop tests failed")
