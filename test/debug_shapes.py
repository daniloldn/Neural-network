#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from neural_network.network import Network


def debug_shapes():
    """Debug the shapes involved in batch processing"""
    print("=" * 60)
    print("DEBUGGING BATCH PROCESSING SHAPES")
    print("=" * 60)

    # Set seed for reproducible results
    np.random.seed(42)

    # Create smaller test case first
    x = np.random.randn(3, 2)  # 3 observations, 2 features
    y = np.random.randn(3, 1)  # 3 target values

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")

    # Create network
    net = Network(x, depth=1, width=2)  # 1 hidden layer, 2 neurons
    print(f"Network: depth={net.depth}, width=2")

    # Get network output
    y_hat = net.feedforward(x)
    print(f"Output shape: {y_hat.shape}")

    # Collect parameters
    params = net._collect_params()

    # Print parameter shapes
    for layer_key, layer_data in params.items():
        print(f"\n{layer_key}:")
        for neuron_key, neuron_data in layer_data.items():
            print(
                f"  {neuron_key}: weights {neuron_data['weights'].shape}, bias scalar"
            )

    # Test _get_z method
    for layer_key in params.keys():
        z_values = net._get_z(params, layer_key)
        print(f"\n{layer_key} z values shape: {z_values.shape}")

    print(f"\nTrying backprop...")
    try:
        deltas = net._backprop(params, y, y_hat)
        print("Backprop successful!")
        for layer_key, delta_values in deltas.items():
            print(f"{layer_key} deltas shape: {delta_values.shape}")
    except Exception as e:
        print(f"Backprop failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    debug_shapes()
