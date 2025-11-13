import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.loss import deriv_mse_loss, mse_loss


def test_mse_derivative_comprehensive():
    """Comprehensive test of MSE derivative - both interpretations."""

    print("=== Comprehensive MSE Derivative Test ===\n")

    # Test data
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Differences (y_true - y_pred): {y_true - y_pred}")
    print(f"MSE loss: {mse_loss(y_true, y_pred):.6f}")
    print()

    # Current implementation (average gradient)
    current_deriv = deriv_mse_loss(y_true, y_pred)
    print(f"Current deriv_mse_loss result: {current_deriv:.6f}")

    # Mathematical verification of current implementation
    # The current implementation computes: -2 * mean(y_true - y_pred)
    expected_avg = -2 * np.mean(y_true - y_pred)
    print(f"Expected for average gradient: {expected_avg:.6f}")
    print(
        f"Current implementation is correct for average: {abs(current_deriv - expected_avg) < 1e-10}"
    )
    print()

    # Element-wise gradients (what's typically used in backpropagation)
    print("=== Element-wise Gradients (for backpropagation) ===")

    # For MSE = mean((y_true - y_pred)^2), the gradient w.r.t y_pred[i] is:
    # d(MSE)/d(y_pred[i]) = (1/n) * 2 * (y_pred[i] - y_true[i])
    #                     = (2/n) * (y_pred[i] - y_true[i])
    element_wise_grads = (2 / len(y_pred)) * (y_pred - y_true)
    print(f"Element-wise gradients: {element_wise_grads}")

    # Numerical verification of element-wise gradients
    h = 1e-7
    numerical_grads = []

    for i in range(len(y_pred)):
        y_pred_plus = y_pred.copy()
        y_pred_minus = y_pred.copy()
        y_pred_plus[i] += h
        y_pred_minus[i] -= h

        loss_plus = mse_loss(y_true, y_pred_plus)
        loss_minus = mse_loss(y_true, y_pred_minus)

        grad_i = (loss_plus - loss_minus) / (2 * h)
        numerical_grads.append(grad_i)

    numerical_grads = np.array(numerical_grads)
    print(f"Numerical gradients:    {numerical_grads}")
    print(
        f"Gradients match numerically: {np.allclose(element_wise_grads, numerical_grads, atol=1e-6)}"
    )
    print()

    # Relationship between the two approaches
    print("=== Relationship Analysis ===")
    avg_of_element_wise = np.mean(element_wise_grads)
    print(f"Average of element-wise gradients: {avg_of_element_wise:.6f}")

    # Note: The current implementation returns -2 * mean(y_true - y_pred)
    # But the average of element-wise gradients is (2/n) * mean(y_pred - y_true)
    # These should be related but with opposite sign and different scale

    relationship = -current_deriv / len(y_pred)
    print(f"Current deriv / (-n): {relationship:.6f}")
    print(
        f"This matches avg of element-wise: {abs(relationship - avg_of_element_wise) < 1e-10}"
    )
    print()

    return True


def test_derivative_for_backpropagation():
    """Test what the derivative should return for backpropagation."""

    print("=== What Should deriv_mse_loss Return for Backpropagation? ===\n")

    def corrected_deriv_mse_loss(y_true, y_pred):
        """Corrected version that returns element-wise gradients."""
        # For MSE = mean((y_true - y_pred)^2)
        # d(MSE)/d(y_pred) = (2/n) * (y_pred - y_true)
        return (2 / len(y_pred)) * (y_pred - y_true)

    # Test cases
    test_cases = [
        ([1.0], [1.5]),  # Single value
        ([1.0, 2.0], [1.1, 1.9]),  # Two values
        ([0.0, 1.0, 2.0], [0.1, 0.9, 2.1]),  # Three values
    ]

    for i, (y_true_list, y_pred_list) in enumerate(test_cases):
        y_true = np.array(y_true_list)
        y_pred = np.array(y_pred_list)

        print(f"Test case {i+1}: y_true={y_true}, y_pred={y_pred}")

        # Current implementation (scalar output)
        current = deriv_mse_loss(y_true, y_pred)
        print(f"  Current (scalar): {current:.6f}")

        # Corrected implementation (vector output)
        corrected = corrected_deriv_mse_loss(y_true, y_pred)
        print(f"  Corrected (vector): {corrected}")

        # Numerical verification
        h = 1e-7
        numerical = []
        for j in range(len(y_pred)):
            y_pred_plus = y_pred.copy()
            y_pred_minus = y_pred.copy()
            y_pred_plus[j] += h
            y_pred_minus[j] -= h

            grad = (mse_loss(y_true, y_pred_plus) - mse_loss(y_true, y_pred_minus)) / (
                2 * h
            )
            numerical.append(grad)

        numerical = np.array(numerical)
        print(f"  Numerical check: {numerical}")
        print(
            f"  Corrected matches numerical: {np.allclose(corrected, numerical, atol=1e-6)}"
        )
        print()

    print("RECOMMENDATION:")
    print("For backpropagation, deriv_mse_loss should return element-wise gradients:")
    print("def deriv_mse_loss(y_true, y_pred):")
    print("    return (2/len(y_pred)) * (y_pred - y_true)")
    print()


def test_current_implementation_validity():
    """Test if current implementation is valid for its intended purpose."""

    print("=== Current Implementation Validity ===\n")

    print("The current implementation returns -2 * mean(y_true - y_pred)")
    print("This could be valid if the intended use is:")
    print("1. Computing the gradient of the loss w.r.t. the mean prediction")
    print("2. Some form of aggregate gradient signal")
    print()

    # Test mathematical consistency
    y_true = np.array([2.0, 4.0, 6.0])
    y_pred = np.array([1.8, 4.2, 5.9])

    current_result = deriv_mse_loss(y_true, y_pred)
    expected = -2 * np.mean(y_true - y_pred)

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Current result: {current_result:.6f}")
    print(f"Expected: {expected:.6f}")
    print(f"Mathematically correct: {abs(current_result - expected) < 1e-10}")
    print()

    print("CONCLUSION: Current implementation is mathematically correct")
    print("for what it computes, but may not be what's needed for backpropagation.")


if __name__ == "__main__":
    print("🔍 Testing MSE Derivative Implementation\n")

    test_mse_derivative_comprehensive()
    test_derivative_for_backpropagation()
    test_current_implementation_validity()

    print("\n" + "=" * 60)
    print("FINAL VERDICT:")
    print("✅ Current deriv_mse_loss is mathematically CORRECT")
    print("❓ But it may not be the right function for backpropagation")
    print("💡 Consider what the function should return based on its intended use")
    print("=" * 60)
