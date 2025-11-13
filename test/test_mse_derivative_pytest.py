import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.loss import deriv_mse_loss, mse_loss


def test_current_mse_derivative_mathematical_correctness():
    """Test that current deriv_mse_loss implementation is mathematically correct."""

    test_cases = [
        ([1.0, 2.0], [1.5, 1.5]),
        ([0.0, 1.0, 2.0], [0.1, 0.9, 2.1]),
        ([5.0], [4.8]),
        ([-1.0, 0.0, 1.0], [-0.5, 0.2, 0.8]),
    ]

    for y_true_list, y_pred_list in test_cases:
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        # Current implementation
        actual = deriv_mse_loss(y_true, y_pred)

        # Mathematical expectation: -2 * mean(y_true - y_pred)
        expected = -2 * np.mean(y_true - y_pred)

        assert abs(actual - expected) < 1e-10, f"Expected {expected}, got {actual}"


def test_mse_derivative_perfect_predictions():
    """Test derivative returns 0 for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    derivative = deriv_mse_loss(y_true, y_pred)
    assert abs(derivative) < 1e-10, "Derivative should be 0 for perfect predictions"


def test_mse_derivative_single_value():
    """Test derivative with single value."""
    y_true = np.array([5.0])
    y_pred = np.array([4.0])

    derivative = deriv_mse_loss(y_true, y_pred)
    expected = -2 * (5.0 - 4.0)  # -2 * 1.0 = -2.0
    assert abs(derivative - expected) < 1e-10


def test_mse_derivative_consistency_with_function_definition():
    """Test that the derivative implements exactly what it claims to implement."""

    # Test multiple random cases
    np.random.seed(42)

    for _ in range(5):
        n_samples = np.random.randint(1, 10)
        y_true = np.random.randn(n_samples)
        y_pred = np.random.randn(n_samples)

        # Current implementation
        actual = deriv_mse_loss(y_true, y_pred)

        # What the current implementation should compute based on its code
        expected = (-2) * (y_true - y_pred).mean()

        assert (
            abs(actual - expected) < 1e-12
        ), "Implementation doesn't match its own definition"


def test_mse_derivative_sign_behavior():
    """Test that derivative has correct sign behavior."""

    # When predictions are too low, derivative should be negative (encouraging increase)
    y_true = np.array([2.0])
    y_pred = np.array([1.0])  # Prediction too low

    derivative = deriv_mse_loss(y_true, y_pred)
    assert derivative < 0, "Derivative should be negative when predictions are too low"

    # When predictions are too high, derivative should be positive (encouraging decrease)
    y_true = np.array([1.0])
    y_pred = np.array([2.0])  # Prediction too high

    derivative = deriv_mse_loss(y_true, y_pred)
    assert derivative > 0, "Derivative should be positive when predictions are too high"


def test_mse_derivative_magnitude_behavior():
    """Test that derivative magnitude relates to error magnitude."""

    y_true = np.array([2.0])

    # Small error
    y_pred_small_error = np.array([2.1])
    deriv_small = abs(deriv_mse_loss(y_true, y_pred_small_error))

    # Large error
    y_pred_large_error = np.array([3.0])
    deriv_large = abs(deriv_mse_loss(y_true, y_pred_large_error))

    assert (
        deriv_large > deriv_small
    ), "Larger errors should produce larger derivative magnitudes"


def test_mse_derivative_symmetry():
    """Test symmetry properties of the derivative."""

    y_true = np.array([2.0, 4.0])

    # Equal positive and negative errors should cancel out
    y_pred_balanced = np.array([1.5, 4.5])  # -0.5 and +0.5 errors

    derivative = deriv_mse_loss(y_true, y_pred_balanced)
    assert (
        abs(derivative) < 1e-10
    ), "Balanced errors should result in zero average derivative"


def test_mse_derivative_scaling():
    """Test how derivative scales with input scaling."""

    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.1, 1.9])

    # Original
    deriv_original = deriv_mse_loss(y_true, y_pred)

    # Scaled by factor of 10
    scale_factor = 10
    deriv_scaled = deriv_mse_loss(y_true * scale_factor, y_pred * scale_factor)

    # Derivative should scale linearly with the scaling factor
    expected_scaled = deriv_original * scale_factor
    assert (
        abs(deriv_scaled - expected_scaled) < 1e-10
    ), "Derivative should scale linearly"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
