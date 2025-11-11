import numpy as np
from typing import Sequence, Union, Callable, Optional
from .activation import sigmoid


class Neuron:
    """a neuron with:
    - sigmoid activation function 
    """
    def __init__(self, n_inputs: int, id: int, activation: Optional[Callable] = None) -> None:
        self.weights = np.random.normal(size=n_inputs)
        self.bias = np.random.normal()
        self.id = id
        self.n_inputs = n_inputs
        self.activation = activation if activation is not None else sigmoid
    
    def feed_forward(self, inputs: Sequence[float]) -> float:
        # compute weighted sum plus bias and apply activation
        x = np.array(inputs)
        z = np.dot(self.weights, x) + self.bias
        return self.activation(z)


class Layer:
    """a layer with n neurons"""
    
    def __init__(self, n_inputs: int, n_neurons: int, activation: Optional[Callable] = None):
        self.neurons = [Neuron(n_inputs, i, activation) for i in range(n_neurons)]

    def feed_forward(self, inputs):
        self.output = [self.neurons[i].feed_forward(inputs) for i in range(len(self.neurons))]
        return self.output
 




