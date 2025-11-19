import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.network import Network


def demonstrate_collect_params():
    """Demonstrate the _collect_params function with a sample network."""

    print("=== _collect_params Function Demonstration ===\n")

    # Create a sample network
    np.random.seed(42)  # For reproducible output
    x = np.array([[1.0, 2.0, 3.0]])  # 3 inputs
    net = Network(x, depth=2, width=2)  # init + 1 hidden + output

    print("Network Architecture:")
    print(f"- Input size: {x.shape[1]}")
    print(f"- Depth: {net.depth} (layers: init + {net.depth-1} hidden + output)")
    print(f"- Width: 2 neurons per hidden layer")
    print(
        f"- Total layers: {len(net.network)} (init_layer, hidden_layer list, output_layer)"
    )

    # Collect parameters
    params = net._collect_params()

    print(f"\n=== Collected Parameters Structure ===")
    print(f"Total layers collected: {len(params)}")
    print(f"Layer keys: {list(params.keys())}")

    # Display detailed structure
    for layer_key, layer_data in params.items():
        print(f"\n{layer_key}:")
        print(f"  Number of neurons: {len(layer_data)}")

        for neuron_key, neuron_data in layer_data.items():
            weights = neuron_data["weights"]
            bias = neuron_data["bias"]
            print(f"    {neuron_key}:")
            print(f"      Weights shape: {weights.shape}")
            print(f"      Weights: {weights.flatten()}")
            print(f"      Bias: {bias:.6f}")

    print(f"\n=== Summary ===")
    total_params = 0
    for layer_data in params.values():
        for neuron_data in layer_data.values():
            weights_count = neuron_data["weights"].size
            bias_count = 1
            total_params += weights_count + bias_count

    print(f"Total parameters in network: {total_params}")
    print(f"Parameters organized by: Layer ID -> Neuron ID -> {{'weights', 'bias'}}")
    print(
        f"Function successfully organizes all network parameters for training algorithms!"
    )


if __name__ == "__main__":
    demonstrate_collect_params()
