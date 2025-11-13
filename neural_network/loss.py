import numpy as np


def mse_loss(y_true: np.array, y_pred: np.array) -> float:
    """
    Compute the mean squared error (MSE) between ground-truth and predicted values.
    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth target values.
    y_pred : np.ndarray
        Predicted values. Must be broadcastable to the shape of `y_true`.
    Returns
    -------
    float
        The mean of the squared differences between `y_true` and `y_pred`:
        mean((y_true - y_pred) ** 2).
    Raises
    ------
    ValueError
        If `y_true` and `y_pred` cannot be broadcast to a common shape.
    Notes
    -----
    This function performs element-wise subtraction and averaging; NaNs or infinities
    in the inputs will propagate to the result.
    Examples
    --------
    >>> mse_loss(np.array([1.0, 2.0]), np.array([0.5, 2.5]))
    0.25
    """
    """mse loss function"""

    return ((y_true - y_pred) ** 2).mean()
