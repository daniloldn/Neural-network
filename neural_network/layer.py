from typing import Callable, Optional

import numpy as np

from .neuron import Neuron


class Layer:
    """Layer representing a collection of neurons that process
    the same input vector.

    Parameters
    ----------
    n_inputs : int
        Number of input signals each neuron expects.
    n_neurons : int
        Number of neurons to create in the layer.
    activation : Optional[Callable], optional
        Activation function to assign to each Neuron. If None, neurons will use
        their default (linear or whatever the Neuron class provides).

    Attributes
    ----------
    neurons : list
        List of Neuron instances created for this layer. Each Neuron is
        constructed with (n_inputs, index, activation) where index ranges from
        0 to n_neurons - 1.
    output : list
        Outputs produced by the most recent call to feed_forward. Each element
        corresponds to the output of one neuron in the layer.

    Methods
    -------
    feed_forward(inputs)
        Compute and return the outputs of all neurons
        given the provided inputs.
        This method also updates the `output` attribute with the computed list.
        Expected `inputs` is an iterable compatible with Neuron.feed_forward
        (typically a sequence of length `n_inputs`).

    Notes
    -----
    - The Layer is a simple container that delegates computation to its Neuron
      objects; it does not perform any aggregation (such as softmax) across its
      neuron outputs.
    - If neurons expect differently shaped inputs or have side effects,
      those behaviors are determined by the Neuron implementation.
    """

    def __init__(
        self, n_inputs: int, n_neurons: int, activation: Optional[Callable] = None
    ):
        self.neurons = [Neuron(n_inputs, i, activation) for i in range(n_neurons)]

    def feed_layer(self, inputs):
        self.output = [
            self.neurons[i].feed_forward(inputs) for i in range(len(self.neurons))
        ]

        if len(self.neurons) > 1:
            # Concatenate all neuron outputs horizontally
            self.out = np.concatenate(self.output, axis=1)
            return self.out

        return self.output[0]
