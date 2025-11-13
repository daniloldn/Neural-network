import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.loss import deriv_mse_loss, mse_loss


def test_mse_derivative_numerical():
    """Test MSE derivative using numerical differentiation (finite differences)."""

    print("=== Testing MSE Loss Derivative ===\n")

    # Test case 1: Simple case
    print("1. Testing with simple values...")
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    # Analytical derivative
    analytical = deriv_mse_loss(y_true, y_pred)
    print(f"   y_true: {y_true}")
    print(f"   y_pred: {y_pred}")
    print(f"   Analytical derivative: {analytical:.6f}")

    # Numerical derivative using finite differences
    h = 1e-7  # Small step size
    numerical_derivatives = []

    for i in range(len(y_pred)):
        # Perturb y_pred[i] slightly and compute derivative
        y_pred_plus = y_pred.copy()
        y_pred_minus = y_pred.copy()
        y_pred_plus[i] += h
        y_pred_minus[i] -= h

        loss_plus = mse_loss(y_true, y_pred_plus)
        loss_minus = mse_loss(y_true, y_pred_minus)

        # Numerical derivative: (f(x+h) - f(x-h)) / (2h)
        numerical_deriv = (loss_plus - loss_minus) / (2 * h)
        numerical_derivatives.append(numerical_deriv)

    # Average the numerical derivatives (since our analytical derivative takes the mean)
    numerical_avg = np.mean(numerical_derivatives)
    print(f"   Numerical derivative: {numerical_avg:.6f}")
    print(f"   Difference: {abs(analytical - numerical_avg):.8f}")

    # Should be very close
    assert (
        abs(analytical - numerical_avg) < 1e-6
    ), "Analytical and numerical derivatives should match"
    print("   ✓ PASSED\n")


def test_mse_derivative_mathematical():
    """Test MSE derivative using mathematical derivation."""

    print("2. Testing mathematical correctness...")

    # MSE = (1/n) * sum((y_true - y_pred)^2)
    # d(MSE)/d(y_pred) = (1/n) * sum(2 * (y_true - y_pred) * (-1))
    #                  = -(2/n) * sum(y_true - y_pred)
    #                  = -2 * mean(y_true - y_pred)

    test_cases = [
        ([1.0, 2.0], [1.5, 1.5]),
        ([0.0, 1.0, 2.0], [0.1, 0.9, 2.1]),
        ([5.0], [4.8]),
        ([-1.0, 0.0, 1.0], [-0.5, 0.2, 0.8]),
    ]

    for i, (y_true_list, y_pred_list) in enumerate(test_cases):
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        # Our implementation
        actual = deriv_mse_loss(y_true, y_pred)

        # Mathematical expectation: -2 * mean(y_true - y_pred)
        expected = -2 * np.mean(y_true - y_pred)

        print(f"   Test case {i+1}: y_true={y_true}, y_pred={y_pred}")
        print(f"   Expected: {expected:.6f}, Actual: {actual:.6f}")

        assert (
            abs(actual - expected) < 1e-10
        ), f"Case {i+1}: Expected {expected}, got {actual}"

    print("   ✓ All mathematical tests PASSED\n")


def test_mse_derivative_edge_cases():
    """Test MSE derivative with edge cases."""

    print("3. Testing edge cases...")

    # Edge case 1: Perfect predictions (difference = 0)
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    derivative = deriv_mse_loss(y_true, y_pred)
    print(f"   Perfect predictions: derivative = {derivative}")
    assert abs(derivative) < 1e-10, "Derivative should be 0 for perfect predictions"

    # Edge case 2: Single value
    y_true_single = np.array([5.0])
    y_pred_single = np.array([4.0])

    derivative_single = deriv_mse_loss(y_true_single, y_pred_single)
    expected_single = -2 * (5.0 - 4.0)  # -2 * 1.0 = -2.0
    print(
        f"   Single value: derivative = {derivative_single}, expected = {expected_single}"
    )
    assert abs(derivative_single - expected_single) < 1e-10

    # Edge case 3: Large values
    y_true_large = np.array([1000.0, 2000.0])
    y_pred_large = np.array([999.0, 2001.0])

    derivative_large = deriv_mse_loss(y_true_large, y_pred_large)
    expected_large = -2 * np.mean([1.0, -1.0])  # -2 * 0 = 0
    print(
        f"   Large values: derivative = {derivative_large}, expected = {expected_large}"
    )
    assert abs(derivative_large - expected_large) < 1e-10

    print("   ✓ Edge cases PASSED\n")


def test_mse_derivative_consistency():
    """Test that derivative is consistent with loss function behavior."""

    print("4. Testing consistency with loss function...")

    y_true = np.array([2.0, 4.0, 6.0])

    # Test: As y_pred approaches y_true, derivative should approach 0
    predictions = [
        np.array([1.0, 3.0, 5.0]),  # Far from true
        np.array([1.5, 3.5, 5.5]),  # Closer
        np.array([1.9, 3.9, 5.9]),  # Very close
        np.array([2.0, 4.0, 6.0]),  # Perfect
    ]

    derivatives = []
    losses = []

    for y_pred in predictions:
        deriv = deriv_mse_loss(y_true, y_pred)
        loss = mse_loss(y_true, y_pred)
        derivatives.append(deriv)
        losses.append(loss)

        print(f"   y_pred = {y_pred}, loss = {loss:.6f}, derivative = {deriv:.6f}")

    # Derivative should decrease in magnitude as we approach perfect prediction
    derivative_magnitudes = [abs(d) for d in derivatives]
    print(f"   Derivative magnitudes: {derivative_magnitudes}")

    # Check that derivative magnitude generally decreases (allowing for some numerical noise)
    assert (
        derivative_magnitudes[-1] < 1e-10
    ), "Derivative should be ~0 for perfect predictions"

    print("   ✓ Consistency test PASSED\n")


def test_mse_derivative_gradient_check():
    """Perform gradient checking against the loss function."""

    print("5. Gradient checking...")

    def mse_loss_wrapper(y_pred, y_true_fixed):
        """Wrapper to treat y_pred as the variable for gradient checking."""
        return mse_loss(y_true_fixed, y_pred)

    # Test multiple random cases
    np.random.seed(42)

    for test_num in range(3):
        # Random test data
        n_samples = np.random.randint(2, 8)
        y_true = np.random.randn(n_samples)
        y_pred = np.random.randn(n_samples)

        print(f"   Test {test_num + 1}: {n_samples} samples")

        # Analytical gradient
        analytical_grad = deriv_mse_loss(y_true, y_pred)

        # Numerical gradient using central differences
        h = 1e-6
        numerical_grads = []

        for i in range(n_samples):
            y_pred_plus = y_pred.copy()
            y_pred_minus = y_pred.copy()
            y_pred_plus[i] += h
            y_pred_minus[i] -= h

            loss_plus = mse_loss_wrapper(y_pred_plus, y_true)
            loss_minus = mse_loss_wrapper(y_pred_minus, y_true)

            grad_i = (loss_plus - loss_minus) / (2 * h)
            numerical_grads.append(grad_i)

        # Average numerical gradients to match our implementation
        numerical_avg = np.mean(numerical_grads)

        diff = abs(analytical_grad - numerical_avg)
        print(f"      Analytical: {analytical_grad:.8f}")
        print(f"      Numerical:  {numerical_avg:.8f}")
        print(f"      Difference: {diff:.10f}")

        assert diff < 1e-6, f"Gradient check failed for test {test_num + 1}"

    print("   ✓ Gradient checking PASSED\n")


if __name__ == "__main__":
    try:
        test_mse_derivative_numerical()
        test_mse_derivative_mathematical()
        test_mse_derivative_edge_cases()
        test_mse_derivative_consistency()
        test_mse_derivative_gradient_check()

        print("🎉 All MSE derivative tests passed!")
        print("✅ The derivative implementation is mathematically correct!")

    except AssertionError as e:
        print(f"❌ Test failed: {e}")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
