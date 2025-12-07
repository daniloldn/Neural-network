# mynet — A Simple Neural Network Implementation

A lightweight feedforward neural network built from scratch for learning purposes. This project implements core neural network concepts—neurons, layers, activation functions, and loss computation—to practice the underlying mathematics, software design, and Python package deployment.

**Note**: This is an educational implementation meant to understand how neural networks work at a fundamental level. For production use, refer to mature frameworks like TensorFlow or PyTorch.

## What It Does

- Simple feedforward neural network with customizable depth and width
- Sigmoid activation function with derivatives
- Mean squared error loss with JAX-based gradient computation
- Basic batch processing utilities
- Type hints and test coverage for learning best practices

## Project Structure

```
mynet/
├── network.py           # Main Network class for building and training models
├── layer.py             # Layer class composing neurons
├── neuron.py            # Single neuron with weights, bias, and activation
├── activation.py        # Sigmoid and derivative functions
├── loss.py              # MSE loss and gradient computation (JAX-based)
├── network_dashboard.py # Utilities for monitoring and visualization
└── __init__.py          # Package initialization
```

## Architecture

### Network
The `Network` class implements a feedforward neural network with:
- **Input layer** (`init_layer`): Processes raw input features
- **Hidden layers** (`hidden_layer`): Configurable depth with consistent width
- **Output layer** (`output_layer`): Produces network predictions

### Layer
A `Layer` is a collection of neurons that process the same input and return per-neuron outputs.

### Neuron
A single `Neuron` computes: `output = activation(weights · inputs + bias)`

### Activation & Loss
- **Sigmoid**: Logistic activation function with derivative support
- **MSE Loss**: Mean squared error with JAX-based automatic differentiation

## Quick Start

### Installation
```bash
pip install -e .
```

### Basic Usage
```python
import numpy as np
from mynet import Network, sigmoid

# Create training data (2 features, 100 samples)
X = np.random.randn(100, 2)

# Build a network: 2 inputs, 1 hidden layer of width 3, 1 output
net = Network(inputs=X, depth=1, width=3, activation=sigmoid)

# Forward pass
predictions = net.feedforward(X)
```

## Dependencies

- **Python**: ≥ 3.10
- **numpy** ≥ 1.23
- **jax** ≥ 0.4.0 (for gradient computation)
- **jaxlib** ≥ 0.4.0
- **matplotlib** ≥ 3.5 (for visualization)
- **sphinx** ≥ 7.0.0 (for documentation)
- **pytest** ≥ 7.0 (for testing)

## Testing

Run the test suite from the repository root:
```bash
pytest -q
```

For type checking:
```bash
mypy mynet/
```

## Development

Install development dependencies:
```bash
pip install -e ".[dev]"
```

This includes:
- **black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **pytest-cov**: Test coverage
- **pre-commit**: Git hooks

## Documentation

Sphinx documentation is available in the `docs/` directory. Build with:
```bash
cd docs
make html
```

## Key Implementation Details

- **Network structure**: `[init_layer, hidden_layer (list), output_layer]`
- **Neuron weights**: Stored as transposed column arrays to optimize dot products
- **Feedforward flow**: init_layer → hidden layers (recursive) → output_layer
- **Loss computation**: Uses JAX for automatic differentiation of MSE loss

## Status

This is an active learning project covering:
- Feedforward propagation fundamentals
- Gradient computation with automatic differentiation
- Package structure and deployment practices
- Testing and type checking

Advanced features like sophisticated training loops, optimizers, and performance optimization are out of scope for this educational project.

## Author

Danilo de Souza

## License

See LICENSE file for details.
