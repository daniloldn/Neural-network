#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.network import Network


def main():
    """Simple test script for update_values function with dummy data"""
    print("Testing update_values function")
    print("=" * 50)

    # Set seed for reproducible results
    np.random.seed(42)

    # Create dummy data
    x = np.array([[1.0, 2.0]])  # Single input sample
    y_true = np.array([[0.5]])  # Target output
    learning_rate = 0.01

    print(f"Input data: {x}")
    print(f"Target: {y_true}")
    print(f"Learning rate: {learning_rate}")

    # Initialize network
    print(f"\nInitializing network...")
    net = Network(x, depth=2, width=2)
    print(f"Network created - Depth: {net.depth}, Width: 2")

    # Forward pass
    print(f"\nForward pass...")
    y_pred = net.feedforward(x)
    print(f"Prediction: {y_pred}")

    # Collect parameters
    print(f"\nCollecting parameters...")
    params = net._collect_params()
    print(f"Parameter layers: {list(params.keys())}")

    # Show current parameters
    print(f"\nCurrent parameters:")
    for layer_key, layer in params.items():
        print(f"  {layer_key}:")
        for neuron_key, neuron in layer.items():
            print(f"    {neuron_key}:")
            print(f"      weights: {neuron['weights'].flatten()}")
            print(f"      bias: {neuron['bias']}")

    # Backpropagation
    print(f"\nBackpropagation...")
    deltas = net._backprop(params, y_true, y_pred)
    print(f"Delta layers: {list(deltas.keys())}")
    for layer_key, delta in deltas.items():
        print(f"  {layer_key}: {delta}")

    # Test update_values
    print(f"\nTesting update_values...")
    try:
        updated_params = net.update_values(x, params, deltas, learning_rate)
        print("✓ update_values completed successfully!")

        # Show updated parameters
        print(f"\nUpdated parameters:")
        for layer_key, layer in updated_params.items():
            print(f"  {layer_key}:")
            for neuron_key, neuron in layer.items():
                print(f"    {neuron_key}:")
                print(f"      weights: {neuron['weights'].flatten()}")
                print(f"      bias: {neuron['bias']}")

    except Exception as e:
        print(f"✗ update_values failed: {e}")
        print(f"\nError details:")
        import traceback

        traceback.print_exc()

        print(f"\nDebugging info:")
        print(f"x shape: {x.shape}")
        print(f"params keys: {list(params.keys())}")
        print(f"deltas keys: {list(deltas.keys())}")
        for key, delta in deltas.items():
            print(f"  {key} delta: shape={delta.shape}, type={type(delta)}")


if __name__ == "__main__":
    main()
