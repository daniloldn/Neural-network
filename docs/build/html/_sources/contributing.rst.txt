Contributing
============

We welcome contributions to the neural network project! This guide will help you get started 
with development, testing, and contributing improvements.

Development Setup
-----------------

1. **Clone the Repository**

   .. code-block:: bash

      git clone https://github.com/daniloldn/Neural-network.git
      cd Neural-network

2. **Set Up Development Environment**

   .. code-block:: bash

      # Create virtual environment
      python -m venv vneural
      source vneural/bin/activate  # On Windows: vneural\Scripts\activate

      # Install in development mode with all dependencies
      pip install -e ".[dev,test,viz]"

3. **Verify Installation**

   .. code-block:: bash

      # Run basic tests
      pytest test/ -v

      # Run deployment tests
      python test/deployment_tests_simple.py

Project Structure
-----------------

::

   Neural-network/
   ├── neural_network/          # Main package
   │   ├── __init__.py
   │   ├── activation.py        # Activation functions
   │   ├── layer.py             # Layer implementation
   │   ├── loss.py              # Loss functions with JAX
   │   ├── network.py           # Main Network class
   │   └── neuron.py            # Individual neuron logic
   ├── test/                    # Test suite
   │   ├── deployment_tests_simple.py
   │   ├── terminal_vis.py
   │   └── various test files
   ├── docs/                    # Documentation
   └── pyproject.toml           # Project configuration

Development Guidelines
----------------------

**Code Style**

We use several tools to maintain code quality:

.. code-block:: bash

   # Format code
   black neural_network/ test/

   # Sort imports
   isort neural_network/ test/

   # Lint code
   flake8 neural_network/ test/

   # Type checking
   mypy neural_network/

**Git Workflow**

1. Create a feature branch: ``git checkout -b feature/your-feature-name``
2. Make your changes
3. Run tests: ``pytest test/``
4. Format code: ``black . && isort .``
5. Commit changes: ``git commit -m "Add: description of changes"``
6. Push branch: ``git push origin feature/your-feature-name``
7. Open a pull request

Testing
-------

**Running Tests**

.. code-block:: bash

   # Run all tests
   pytest test/ -v

   # Run specific test file
   pytest test/network_test.py -v

   # Run tests with coverage
   pytest test/ --cov=neural_network --cov-report=html

**Test Categories**

* **Unit Tests**: Test individual components (neurons, layers, functions)
* **Integration Tests**: Test component interactions
* **Deployment Tests**: Test on realistic data patterns
* **Mathematical Tests**: Verify gradient computations and loss calculations

**Writing New Tests**

Follow the existing test patterns:

.. code-block:: python

   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

   import numpy as np
   import pytest
   from neural_network import Network

   def test_network_creation():
       """Test basic network creation."""
       X = np.random.randn(10, 2)
       net = Network(X, depth=1, width=3)
       assert net.depth == 1
       assert len(net.init_layer.neurons) == 3

   def test_feedforward_shape():
       """Test that feedforward returns correct shape."""
       X = np.random.randn(10, 2)
       net = Network(X, depth=1, width=3)
       output = net.feedforward(X)
       assert output.shape == (10, 1)

Areas for Contribution
----------------------

**High Priority**

* **Performance Optimization**: Improve training speed and convergence
* **Additional Activations**: ReLU, tanh, leaky ReLU implementations
* **Advanced Optimizers**: Adam, RMSprop, momentum-based optimizers
* **Regularization**: L1/L2 regularization, dropout implementation
* **GPU Support**: CUDA acceleration through JAX

**Medium Priority**

* **Network Architectures**: Convolutional layers, LSTM/RNN support
* **Loss Functions**: Cross-entropy, custom loss function support
* **Visualization**: Enhanced plotting, network architecture diagrams
* **Data Handling**: Built-in data preprocessing, normalization utilities
* **Serialization**: Save/load trained models

**Documentation & Examples**

* **Tutorials**: Step-by-step guides for common use cases
* **API Documentation**: Improved docstrings and examples
* **Mathematical Background**: Theory explanations and derivations
* **Performance Benchmarks**: Comparison with other frameworks

Code Review Checklist
----------------------

Before submitting a pull request, ensure:

**Functionality**
   ☐ Code runs without errors
   ☐ New features work as intended
   ☐ Existing functionality is not broken
   ☐ Edge cases are handled appropriately

**Testing**
   ☐ New code has corresponding tests
   ☐ All tests pass locally
   ☐ Test coverage is maintained or improved
   ☐ Tests are meaningful and thorough

**Code Quality**
   ☐ Code follows project style (black, isort)
   ☐ No lint warnings (flake8)
   ☐ Type hints are present (mypy)
   ☐ Docstrings follow NumPy/Google style

**Documentation**
   ☐ Public functions have docstrings
   ☐ Complex logic is commented
   ☐ README/docs updated if needed
   ☐ Examples work as written

Common Development Tasks
------------------------

**Adding a New Activation Function**

1. Add function to ``neural_network/activation.py``:

   .. code-block:: python

      def relu(x):
          """ReLU activation function."""
          return np.maximum(0, x)

      def deriv_relu(x):
          """Derivative of ReLU."""
          return (x > 0).astype(float)

2. Update imports and usage in relevant files
3. Add tests in ``test/activation_test.py``
4. Update documentation

**Adding a New Optimizer**

1. Create ``neural_network/optimizers.py``
2. Implement optimizer class with ``update`` method
3. Integrate into ``Network.train()`` method
4. Add comprehensive tests
5. Update documentation and examples

**Performance Profiling**

.. code-block:: bash

   # Profile training performance
   python -m cProfile -o profile_output.prof test/deployment_tests_simple.py

   # Analyze results
   python -c "
   import pstats
   p = pstats.Stats('profile_output.prof')
   p.sort_stats('cumulative').print_stats(20)
   "

Debugging Tips
--------------

**Common Issues**

* **Import Errors**: Ensure you're using ``sys.path.insert(0, ...)`` in test files
* **Shape Mismatches**: Check array dimensions in forward/backward passes
* **Slow Convergence**: Verify gradient calculations and learning rates
* **Memory Issues**: Use smaller batch sizes or reduce network complexity

**Debugging Tools**

.. code-block:: python

   # Check gradients numerically
   from neural_network.loss import mse_loss
   import jax

   def check_gradients(net, X, y):
       """Verify gradient calculations."""
       grad_fn = jax.grad(lambda params: mse_loss(y, net.feedforward(X)))
       # Compare with numerical gradients

   # Monitor training progress
   import matplotlib.pyplot as plt

   def plot_training_history(losses):
       """Plot loss over time."""
       plt.plot(losses)
       plt.title('Training Loss')
       plt.xlabel('Epoch')
       plt.ylabel('Loss')
       plt.show()

Getting Help
------------

* **GitHub Issues**: Report bugs or request features
* **Documentation**: Check examples and API reference
* **Code Comments**: Most functions have detailed explanations
* **Test Files**: See test/ directory for usage examples

**Before Opening an Issue**

1. Check existing issues for duplicates
2. Provide minimal reproducible example
3. Include error messages and tracebacks
4. Specify your environment (OS, Python version, dependencies)

Thank you for contributing to the neural network project! Your improvements help 
make this educational tool better for everyone.