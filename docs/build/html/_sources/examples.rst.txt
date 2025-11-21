Examples
========

This section provides practical examples of using the neural network implementation
for various machine learning tasks.

Basic Usage Example
-------------------

Here's a simple example to get you started:

.. code-block:: python

   import numpy as np
   from neural_network import Network

   # Create synthetic data
   np.random.seed(42)
   X = np.random.uniform(-1, 1, (100, 2))  # 100 samples, 2 features
   y = (X[:, 0] + X[:, 1]).reshape(-1, 1)  # Sum of features

   # Create network
   net = Network(X, depth=1, width=3)

   # Train the network
   net.train(X, y, epochs=30, batch_size=16, learning_rate=0.1)

   # Make predictions
   predictions = net.feedforward(X)
   print(f"Final loss: {np.mean((y - predictions)**2):.6f}")

Linear Regression Example
-------------------------

Train a neural network to learn a linear relationship:

.. code-block:: python

   import numpy as np
   from neural_network import Network
   from neural_network.loss import mse_loss

   # Generate linear data: y = 2*x1 + x2 + noise
   np.random.seed(42)
   n_samples = 80
   X = np.random.uniform(-1, 1, (n_samples, 2))
   y = 2 * X[:, 0] + X[:, 1] + 0.1 * np.random.randn(n_samples)
   y = y.reshape(-1, 1)

   # Split into train/test
   split_idx = int(0.8 * len(X))
   X_train, X_test = X[:split_idx], X[split_idx:]
   y_train, y_test = y[:split_idx], y[split_idx:]

   # Create and train network
   net = Network(X_train, depth=1, width=3)
   net.train(X_train, y_train, epochs=40, batch_size=16, learning_rate=0.3)

   # Evaluate performance
   train_pred = net.feedforward(X_train)
   test_pred = net.feedforward(X_test)

   train_loss = mse_loss(y_train, train_pred)
   test_loss = mse_loss(y_test, test_pred)

   print(f"Training Loss: {train_loss:.6f}")
   print(f"Test Loss: {test_loss:.6f}")

Non-linear Pattern Example
---------------------------

Train on quadratic data to test non-linear learning:

.. code-block:: python

   import numpy as np
   from neural_network import Network

   # Generate quadratic data: y = x^2
   np.random.seed(42)
   x = np.random.uniform(-2, 2, 100)
   X = np.column_stack([x, x**2])  # Features: x and x^2
   y = x**2 + 0.1 * np.random.randn(100)
   y = y.reshape(-1, 1)

   # Create deeper network for non-linear patterns
   net = Network(X, depth=2, width=4)
   net.train(X, y, epochs=60, batch_size=16, learning_rate=0.2)

   # Check a few predictions
   test_x = np.array([[-1], [0], [1], [2]])
   test_X = np.column_stack([test_x.flatten(), test_x.flatten()**2])
   predictions = net.feedforward(test_X)

   print("x     | Predicted | Expected")
   print("------|-----------|----------")
   for i, x_val in enumerate(test_x.flatten()):
       pred = predictions[i, 0]
       expected = x_val**2
       print(f"{x_val:5.1f} | {pred:9.3f} | {expected:8.1f}")

XOR Pattern Example
-------------------

Attempt to learn the classic XOR pattern (challenging for neural networks):

.. code-block:: python

   import numpy as np
   from neural_network import Network

   # Generate XOR-like data
   np.random.seed(42)
   n_per_cluster = 40

   # Four clusters representing XOR pattern
   # Bottom-left and top-right: output 0
   # Top-left and bottom-right: output 1

   x1 = np.concatenate([
       np.random.normal(-1, 0.3, n_per_cluster),  # bottom-left
       np.random.normal(1, 0.3, n_per_cluster),   # top-right
       np.random.normal(-1, 0.3, n_per_cluster),  # top-left
       np.random.normal(1, 0.3, n_per_cluster)    # bottom-right
   ])

   x2 = np.concatenate([
       np.random.normal(-1, 0.3, n_per_cluster),  # bottom-left
       np.random.normal(1, 0.3, n_per_cluster),   # top-right
       np.random.normal(1, 0.3, n_per_cluster),   # top-left
       np.random.normal(-1, 0.3, n_per_cluster)   # bottom-right
   ])

   X = np.column_stack([x1, x2])
   y = np.concatenate([
       np.zeros(n_per_cluster),  # bottom-left: 0
       np.zeros(n_per_cluster),  # top-right: 0
       np.ones(n_per_cluster),   # top-left: 1
       np.ones(n_per_cluster)    # bottom-right: 1
   ]).reshape(-1, 1)

   # Create network with sufficient capacity for XOR
   net = Network(X, depth=2, width=8)
   net.train(X, y, epochs=100, batch_size=16, learning_rate=0.1)

   # Test on the four XOR corners
   test_points = np.array([[-1, -1], [1, 1], [-1, 1], [1, -1]])
   predictions = net.feedforward(test_points)

   print("XOR Test Results:")
   print("x1    x2   | Predicted | Expected")
   print("-----------|-----------|----------")
   expected = [0, 0, 1, 1]
   for i, (x1, x2) in enumerate(test_points):
       pred = predictions[i, 0]
       exp = expected[i]
       print(f"{x1:4.0f} {x2:4.0f}  | {pred:9.3f} | {exp:8.0f}")

Deployment Testing Example
--------------------------

Run comprehensive tests across multiple data patterns:

.. code-block:: python

   # Run the deployment test suite
   exec(open('test/deployment_tests_simple.py').read())

This will test the network on:

* Linear relationships (simple regression)
* Quadratic patterns (non-linear fitting)
* Sine waves (complex oscillatory patterns)
* XOR patterns (non-linear classification boundaries)

Each test provides detailed performance metrics and terminal visualization.

Custom Training Loop Example
-----------------------------

For more control over the training process:

.. code-block:: python

   import numpy as np
   from neural_network import Network
   from neural_network.loss import mse_loss

   # Generate data
   X = np.random.randn(50, 2)
   y = np.sum(X, axis=1, keepdims=True)

   # Create network
   net = Network(X, depth=2, width=4)

   # Custom training with monitoring
   losses = []
   for epoch in range(30):
       # Forward pass
       y_pred = net.feedforward(X)
       
       # Calculate loss
       loss = mse_loss(y, y_pred)
       losses.append(float(loss))
       
       # Training step (simplified - normally done in batches)
       net.train(X, y, epochs=1, batch_size=len(X), learning_rate=0.1)
       
       # Print progress
       if epoch % 10 == 0:
           print(f"Epoch {epoch}: Loss = {loss:.6f}")

   print(f"Final loss: {losses[-1]:.6f}")
   print(f"Total improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

Performance Tips
----------------

**For Better Learning:**

1. **Start Simple**: Begin with shallow networks (depth=1) and small widths
2. **Learning Rate**: Try values between 0.01 and 0.5, adjust based on convergence
3. **Batch Size**: Smaller batches (8-32) often work better than large ones
4. **Epochs**: Start with 30-50 epochs, increase if still improving
5. **Data Scaling**: Normalize inputs to [-1, 1] or [0, 1] range

**Common Issues:**

* **Slow Learning**: Increase learning rate or decrease network complexity
* **No Improvement**: Check data normalization and learning rate
* **Unstable Training**: Reduce learning rate or batch size
* **Poor Generalization**: Reduce network size or increase training data

**Debugging Tools:**

.. code-block:: python

   # Check network structure
   print(f"Network depth: {net.depth}")
   print(f"Input features: {net.inputs.shape[1]}")
   
   # Monitor weight changes
   initial_weights = net.init_layer.neurons[0].weights.copy()
   # ... after training ...
   final_weights = net.init_layer.neurons[0].weights
   weight_change = np.mean(np.abs(final_weights - initial_weights))
   print(f"Average weight change: {weight_change:.6f}")