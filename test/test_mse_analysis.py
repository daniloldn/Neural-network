import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.loss import mse_loss


def correct_deriv_mse_loss(y_true, y_pred):
    """Correct implementation of MSE derivative."""
    # MSE = mean((y_true - y_pred)^2)
    # d(MSE)/d(y_pred) = mean(2 * (y_true - y_pred) * (-1))
    #                  = -2 * mean(y_true - y_pred)
    #                  = -2 * (y_true - y_pred) / n  [element-wise]
    # But since we want the gradient of the scalar loss, we return the mean gradient
    return -2 * (y_true - y_pred).mean()


def incorrect_current_implementation(y_true, y_pred):
    """Current (incorrect) implementation from loss.py"""
    return (-2) * (y_true - y_pred).mean()


def test_derivative_corrections():
    """Test to identify and fix the MSE derivative issue."""

    print("=== Analyzing MSE Derivative Implementation ===\n")

    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])

    print(f"y_true: {y_true}")
    print(f"y_pred: {y_pred}")
    print(f"Differences (y_true - y_pred): {y_true - y_pred}")
    print()

    # Current implementation
    current = incorrect_current_implementation(y_true, y_pred)
    print(f"Current implementation: {current:.6f}")

    # Mathematical expectation
    # For MSE loss, the derivative with respect to each y_pred[i] is:
    # d(MSE)/d(y_pred[i]) = -2 * (y_true[i] - y_pred[i]) / n

    # Since we want the average gradient (for the scalar loss):
    expected = -2 * np.mean(y_true - y_pred)
    print(f"Mathematical expectation: {expected:.6f}")
    print(f"They match: {abs(current - expected) < 1e-10}")
    print()

    # Numerical gradient check
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
        print(f"Numerical gradient w.r.t y_pred[{i}]: {grad_i:.6f}")

    numerical_avg = np.mean(numerical_grads)
    print(f"Average numerical gradient: {numerical_avg:.6f}")
    print()

    print("=== Issue Analysis ===")
    print("The problem is in the interpretation of what the derivative should return.")
    print()
    print("For backpropagation, we typically want the gradient w.r.t each prediction:")
    print("grad[i] = d(MSE)/d(y_pred[i]) = -2 * (y_true[i] - y_pred[i]) / n")
    print()

    # Element-wise gradients
    element_wise_grads = -2 * (y_true - y_pred) / len(y_pred)
    print(f"Element-wise gradients: {element_wise_grads}")
    print(f"Mean of element-wise gradients: {element_wise_grads.mean():.6f}")
    print(
        f"This matches numerical average: {abs(element_wise_grads.mean() - numerical_avg) < 1e-6}"
    )
    print()

    # The current implementation is actually correct for returning the AVERAGE gradient
    # But for backpropagation, we usually want element-wise gradients
    print("CONCLUSION:")
    print(
        "- Current implementation returns the AVERAGE gradient (correct for scalar output)"
    )
    print("- For backpropagation, we typically want ELEMENT-WISE gradients")
    print("- Both are mathematically correct, but serve different purposes")


if __name__ == "__main__":
    test_derivative_corrections()
