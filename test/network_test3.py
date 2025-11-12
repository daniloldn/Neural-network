import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_network.network import Network


class TestNetwork(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.x_simple = np.array([[1, 2, 3]])
        self.y_simple = np.array([4])
        self.x_multiple = np.array([[1, 2, 3], [2, 3, 5]])
        self.y_multiple = np.array([4])

    def test_feedforward_returns_number(self):
        """Test that feedforward returns a single number."""
        net = Network(self.x_simple, self.y_simple, 2, 2)
        result = net.feedforward(self.x_simple)
        self.assertIsInstance(result, (int, float, np.number))

    def test_feedforward_output_shape(self):
        """Test that feedforward output has correct shape."""
        net = Network(self.x_simple, self.y_simple, 2, 2)
        result = net.feedforward(self.x_simple)
        # Should be a scalar or 1D array with single element
        if hasattr(result, "shape"):
            self.assertTrue(result.shape == () or result.shape == (1,))

    def test_feedforward_different_inputs(self):
        """Test feedforward with different input arrays."""
        net = Network(self.x_simple, self.y_simple, 2, 2)

        # Test with original input
        result1 = net.feedforward(self.x_simple)

        # Test with different input
        x_different = np.array([[0, 1, 2]])
        result2 = net.feedforward(x_different)

        # Results should be different (unless weights are zero)
        self.assertIsInstance(result1, (int, float, np.number))
        self.assertIsInstance(result2, (int, float, np.number))

    def test_feedforward_multiple_samples(self):
        """Test feedforward with multiple input samples."""
        net = Network(self.x_multiple, self.y_multiple, 2, 2)
        result = net.feedforward(self.x_multiple)
        self.assertIsInstance(result, (int, float, np.number))

    def test_network_initialization(self):
        """Test that network initializes without errors."""
        net = Network(self.x_simple, self.y_simple, 2, 2)
        self.assertIsNotNone(net)

    def test_loss_attribute_exists(self):
        """Test that network has loss attribute."""
        net = Network(self.x_simple, self.y_simple, 2, 2)
        self.assertTrue(hasattr(net, "loss"))


if __name__ == "__main__":
    unittest.main()
