Neural Network Documentation
============================

A custom neural network implementation built from scratch with JAX-based automatic differentiation, 
comprehensive terminal-based visualization, and modular architecture for educational and research purposes.

.. image:: https://img.shields.io/badge/python-3.10%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: License

Overview
--------

This neural network implementation provides:

* **Modular Architecture**: Clear separation between neurons, layers, and networks
* **JAX Integration**: Automatic differentiation for gradient computation
* **Terminal Visualization**: Beautiful ASCII-based training progress monitoring
* **Educational Focus**: Well-documented code for learning neural network internals
* **Flexible Configuration**: Support for various architectures and hyperparameters

Quick Start
-----------

.. code-block:: python

   import numpy as np
   from neural_network import Network

   # Generate sample data
   X = np.random.randn(100, 3)  # 100 samples, 3 features
   y = np.sum(X, axis=1, keepdims=True)  # Simple sum target

   # Create and train network
   net = Network(X, depth=2, width=5)
   net.train(X, y, epochs=50, batch_size=16, learning_rate=0.1)

   # Make predictions
   predictions = net.feedforward(X)

Key Features
------------

🧠 **Modular Design**
   * ``Neuron``: Individual processing units with weights and biases
   * ``Layer``: Collections of neurons with coordinated forward propagation
   * ``Network``: Complete neural network with training capabilities

📊 **Advanced Visualization**
   * Real-time terminal dashboard with training metrics
   * ASCII loss curves and progress tracking
   * No external GUI dependencies required

🔬 **Educational Value**
   * Clear implementation of backpropagation algorithm
   * Well-documented mathematical operations
   * Comprehensive test suite with various data patterns

⚙️ **Technical Highlights**
   * JAX-based automatic differentiation
   * Batch processing with proper gradient averaging
   * Flexible activation functions (sigmoid with derivatives)
   * MSE loss with custom derivative computation

Architecture Overview
---------------------

The neural network follows a hierarchical structure:

.. code-block:: text

   Network
   ├── Initial Layer (input → first hidden)
   ├── Hidden Layers (configurable depth)
   └── Output Layer (final hidden → output)

Each layer contains multiple neurons that:

1. Receive weighted inputs
2. Apply activation function (sigmoid)
3. Pass outputs to next layer
4. Update weights via backpropagation

Training Process
----------------

The training pipeline implements:

1. **Forward Pass**: Data flows through network layers
2. **Loss Calculation**: MSE between predictions and targets
3. **Backward Pass**: Gradients computed via JAX autodiff
4. **Parameter Update**: Weights and biases adjusted
5. **Visualization**: Progress displayed in terminal

Installation & Dependencies
---------------------------

Install the package from PyPI:

.. code-block:: bash

   # Core installation
   pip install mynet

   # For development (clone repo first)
   git clone <repository-url>
   cd Neural_network
   pip install -e ".[dev]"

Dependencies:
   * NumPy ≥1.23 (numerical computations)
   * JAX ≥0.4.0 (automatic differentiation)
   * Matplotlib ≥3.5 (optional plotting)
   * Pytest ≥7.0 (testing framework)

Testing & Validation
--------------------

The implementation includes comprehensive tests:

* **Unit Tests**: Individual component validation
* **Integration Tests**: End-to-end workflow testing
* **Deployment Tests**: Real-world data pattern validation
* **Mathematical Verification**: Gradient and loss computation accuracy

Run the test suite:

.. code-block:: bash

   # Run all tests
   pytest test/

   # Run deployment validation
   python test/deployment_tests_simple.py

Performance Characteristics
---------------------------

**Strengths:**
   * Mathematically correct implementation
   * Stable training with proper convergence
   * Excellent visualization capabilities
   * Educational value and code clarity

**Current Limitations:**
   * Slow convergence on complex patterns
   * Limited to sigmoid activation
   * CPU-only computation (no GPU acceleration)
   * Fixed architecture during training

**Recommended Use Cases:**
   * Educational demonstrations
   * Algorithm research and experimentation  
   * Small to medium datasets (<1000 samples)
   * Proof-of-concept neural network applications

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   ../../modules

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples

.. toctree::
   :maxdepth: 2
   :caption: Development:

   contributing
   changelog

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

