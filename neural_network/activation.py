import numpy as np


def sigmoid(x: float) -> float:
    """
    Compute the logistic sigmoid activation function.

    Parameters
    ----------
    x : float or array-like
        Input scalar or array of values.

    Returns
    -------
    float or numpy.ndarray
        The sigmoid applied elementwise: 1 / (1 + exp(-x)).
        For scalar input returns a scalar
        float; for array-like input returns a numpy array with the same shape.

    Notes
    -----
    This implementation uses numpy.exp and therefore requires numpy
    to be imported as `np`.
    For very large magnitude inputs, numerical overflow/underflow can
    occur; for improved
    stability use a numerically stable implementation
    (e.g., clamping or using expm1).

    Examples
    --------
    >>> import numpy as np
    >>> sigmoid(0.0)
    0.5
    >>> sigmoid(np.array([-1.0, 0.0, 1.0]))
    array([0.26894142, 0.5, 0.73105858])
    """
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x: float) -> float:
    """
    Compute the derivative of the sigmoid activation function
    at a given input.

    Parameters
    ----------
    x : float
        The input value at which to evaluate the derivative of the
        sigmoid function.

    Returns
    -------
    float
        The value of the derivative
        d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)).

    Notes
    -----
    This implementation calls sigmoid(x) internally to obtain sigmoid(x) and then
    computes sigmoid(x) * (1 - sigmoid(x)). If the sigmoid value is already
    available, pass that value instead (or refactor) to avoid recomputing it.
    """
    fx = sigmoid(x)
    return fx * (1 - fx)
