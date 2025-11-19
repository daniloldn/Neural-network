from typing import Callable, Optional, Sequence, Union

import numpy as np

from .activation import sigmoid


class Neuron:
    """Represent a single artificial neuron with a configurable activation function.

    This class models a basic feed-forward neuron that computes a weighted sum
    of its inputs, adds a bias, and passes the result through an activation
    function.

    Parameters
    ----------
    n_inputs : int
        Number of inputs the neuron expects. Used to size the weight vector.
    id : int
        Identifier for the neuron instance (useful for debugging or tracking).
    activation : Optional[Callable[[float], float]], optional
        Activation function applied to the neuron's pre-activation value
        (weighted sum + bias). If None, a sigmoid function is used by default.

    Attributes
    ----------
    weights : numpy.ndarray
        1-D array of length ``n_inputs`` containing the neuron's weights.
        Initialized randomly (normal distribution) on construction.
    bias : float
        Scalar bias term for the neuron. Initialized randomly (normal distribution).
    id : int
        The identifier provided at construction.
    n_inputs : int
        The number of inputs the neuron expects.
    activation : Callable[[float], float]
        The activation function used to compute the neuron's output.

    Methods
    -------
    feed_forward(inputs: Sequence[float]) -> float
        Compute the neuron's output for a given sequence of input values by
        calculating the dot product of weights and inputs, adding the bias,
        and applying the activation function.

    Notes
    -----
    - The input sequence passed to ``feed_forward`` must have length equal to
      ``n_inputs``; otherwise a ValueError is raised.
    - Weights and bias are initialized using a normal distribution to break
      symmetry for training in multi-neuron networks.

    Examples
    --------
    >>> neuron = Neuron(n_inputs=3, id=1)
    >>> out = neuron.feed_forward([0.1, -0.2, 0.3])
    """

    def __init__(
        self, n_inputs: int, id: int, activation: Optional[Callable] = None
    ) -> None:
        """Initialize a Neuron with given number of inputs, identifier, and activation function."""

        self.weights = np.transpose(np.array([np.random.normal(size=n_inputs)]))
        self.bias = np.random.normal()
        self.id = id
        self.n_inputs = n_inputs
        self.activation = activation if activation is not None else sigmoid

    def feed_forward(self, inputs: Sequence[float]) -> float:
        """Compute the neuron's output for given inputs
        compute weighted sum plus bias and apply activation."""

        x = np.array(inputs)
        z = np.dot(x, self.weights) + self.bias
        self.z = z
        return self.activation(z)
