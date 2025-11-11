import numpy as np
from typing import Sequence, Union

def sigmoid(x: float)-> float:
    return 1 / (1 + np.exp(-x))

class Neuron:
    """a neuron with:
    - sigmoid activation function 
    """
    def __init__(self, weight: np.ndarray, bias: float) -> None:
        self.weight: np.ndarray = weight
        self.bias: float = bias

    def feed_forward(self, inputs: Union[np.ndarray, Sequence[float]]) -> Union[float, np.ndarray]:
        total = np.dot(self.weight, np.array(inputs)) + self.bias
        return sigmoid(total)




