import numpy as np
from layer import Layer


class Network:
    """A neural network with:
    - n inputs
    - h hidden layers
    - y outputs"""

    def __init__(self, inputs, output, depth, width, activation, loss):

        self.loss = loss
        self.init_layer = Layer((inputs.shape[1]), width, activation)
        self.hidden_layer = [Layer(width, width, activation) for _ in range(depth - 1)]
        self.output_layer = Layer(width, output.shape[1], activation)

    def feedforwad(self, x):
        inputs = x.shape[1]
        d = self.depth
        q = self.width
        weights = [np.random.normal() for i in range((inputs * d * q) + q)]

        # need to add layers now that the Layers class is made and working
