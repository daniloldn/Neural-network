import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.loss import deriv_mse_loss, mse_loss


def test_jax_gradient_basic_functionality():
    """Test that JAX gradient computation runs without errors."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 1.8, 2.9])

    # Should not raise any exceptions
    gradient = deriv_mse_loss(y_true, y_pred)

    # Should return a numpy array-like object
    assert hasattr(gradient, "shape"), "Gradient should have shape attribute"
    assert (
        gradient.shape == y_pred.shape
    ), f"Gradient shape {gradient.shape} should match predictions shape {y_pred.shape}"


def test_jax_gradient_mathematical_correctness():
    """Test that JAX gradient matches analytical gradient of MSE."""
    test_cases = [
        ([1.0, 2.0], [1.5, 1.5]),
        ([0.0, 1.0, 2.0], [0.1, 0.9, 2.1]),
        ([5.0], [4.8]),
        ([-1.0, 0.0, 1.0], [-0.5, 0.2, 0.8]),
        ([2.5, 3.1, 1.7], [2.0, 3.5, 1.2]),
    ]

    for y_true_list, y_pred_list in test_cases:
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        # JAX gradient
        jax_gradient = deriv_mse_loss(y_true, y_pred)

        # Analytical gradient of MSE w.r.t. y_pred: 2 * (y_pred - y_true) / n
        n = len(y_true)
        analytical_gradient = 2 * (y_pred - y_true) / n

        # Should match within numerical precision (relaxed for float32)
        np.testing.assert_allclose(
            jax_gradient,
            analytical_gradient,
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"JAX gradient {jax_gradient} != analytical {analytical_gradient}",
        )


def test_jax_gradient_perfect_predictions():
    """Test gradient is zero for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0, -1.0, 0.0])
    y_pred = y_true.copy()  # Perfect predictions

    gradient = deriv_mse_loss(y_true, y_pred)

    # Should be all zeros
    np.testing.assert_allclose(
        gradient,
        np.zeros_like(y_true),
        rtol=1e-12,
        atol=1e-15,
        err_msg="Gradient should be zero for perfect predictions",
    )


def test_jax_gradient_single_value():
    """Test gradient computation with single value."""
    y_true = np.array([5.0])
    y_pred = np.array([4.0])

    gradient = deriv_mse_loss(y_true, y_pred)

    # Expected: 2 * (4.0 - 5.0) / 1 = -2.0
    expected = np.array([-2.0])

    np.testing.assert_allclose(
        gradient,
        expected,
        rtol=1e-10,
        atol=1e-12,
        err_msg=f"Single value gradient {gradient} != expected {expected}",
    )


def test_jax_gradient_sign_behavior():
    """Test gradient has correct sign for over/under predictions."""
    # Overprediction case
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([2.0, 3.0])  # All predictions too high

    gradient = deriv_mse_loss(y_true, y_pred)

    # Gradient should be positive (gradient points in direction to increase loss)
    assert np.all(
        gradient > 0
    ), f"Gradient should be positive for overpredictions, got {gradient}"

    # Underprediction case
    y_pred = np.array([0.5, 1.0])  # All predictions too low

    gradient = deriv_mse_loss(y_true, y_pred)

    # Gradient should be negative (gradient points in direction to increase loss)
    assert np.all(
        gradient < 0
    ), f"Gradient should be negative for underpredictions, got {gradient}"


def test_jax_gradient_magnitude_scaling():
    """Test gradient magnitude scales with prediction error."""
    y_true = np.array([5.0])

    # Small error
    y_pred_small = np.array([5.1])
    grad_small = deriv_mse_loss(y_true, y_pred_small)

    # Large error
    y_pred_large = np.array([6.0])
    grad_large = deriv_mse_loss(y_true, y_pred_large)

    # Larger error should give larger gradient magnitude
    assert abs(grad_large[0]) > abs(
        grad_small[0]
    ), f"Larger error should give larger gradient: {abs(grad_large[0])} > {abs(grad_small[0])}"


def test_jax_gradient_numerical_stability():
    """Test gradient computation with extreme values."""
    # Very small values
    y_true = np.array([1e-8, 1e-7])
    y_pred = np.array([2e-8, 3e-7])

    gradient = deriv_mse_loss(y_true, y_pred)
    assert np.all(np.isfinite(gradient)), "Gradient should be finite for small values"

    # Large values
    y_true = np.array([1e6, 1e7])
    y_pred = np.array([1.1e6, 1.2e7])

    gradient = deriv_mse_loss(y_true, y_pred)
    assert np.all(np.isfinite(gradient)), "Gradient should be finite for large values"


def test_jax_gradient_consistency_with_mse():
    """Test that gradient is consistent with MSE loss function."""
    y_true = np.array([2.0, 3.0, 1.0])
    y_pred = np.array([2.1, 2.8, 1.3])

    # Compute gradient
    gradient = deriv_mse_loss(y_true, y_pred)

    # Numerical gradient check using finite differences
    h = 1e-7  # Small step size
    numerical_gradient = np.zeros_like(y_pred)

    for i in range(len(y_pred)):
        # f(x + h)
        y_pred_plus = y_pred.copy()
        y_pred_plus[i] += h
        loss_plus = mse_loss(y_true, y_pred_plus)

        # f(x - h)
        y_pred_minus = y_pred.copy()
        y_pred_minus[i] -= h
        loss_minus = mse_loss(y_true, y_pred_minus)

        # Numerical gradient: (f(x+h) - f(x-h)) / (2*h)
        numerical_gradient[i] = (loss_plus - loss_minus) / (2 * h)

    # JAX gradient should match numerical gradient
    np.testing.assert_allclose(
        gradient,
        numerical_gradient,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"JAX gradient {gradient} doesn't match numerical gradient {numerical_gradient}",
    )


def test_jax_gradient_different_array_types():
    """Test gradient computation with different numpy array types."""
    y_true_list = [1.0, 2.0, 3.0]
    y_pred_list = [1.2, 1.8, 3.1]

    # Test with different input types
    test_types = [
        (np.array(y_true_list), np.array(y_pred_list)),
        (
            np.array(y_true_list, dtype=np.float32),
            np.array(y_pred_list, dtype=np.float32),
        ),
        (
            np.array(y_true_list, dtype=np.float64),
            np.array(y_pred_list, dtype=np.float64),
        ),
    ]

    reference_gradient = None

    for y_true, y_pred in test_types:
        gradient = deriv_mse_loss(y_true, y_pred)

        if reference_gradient is None:
            reference_gradient = gradient
        else:
            # All should give essentially the same result
            np.testing.assert_allclose(
                gradient,
                reference_gradient,
                rtol=1e-6,
                atol=1e-9,
                err_msg=f"Gradient should be consistent across dtypes",
            )


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_jax_gradient_basic_functionality,
        test_jax_gradient_mathematical_correctness,
        test_jax_gradient_perfect_predictions,
        test_jax_gradient_single_value,
        test_jax_gradient_sign_behavior,
        test_jax_gradient_magnitude_scaling,
        test_jax_gradient_numerical_stability,
        test_jax_gradient_consistency_with_mse,
        test_jax_gradient_different_array_types,
    ]

    print("Running JAX gradient tests...")
    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")

    print("All tests completed!")
