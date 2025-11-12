import numpy as np
from neuron import Neuron


class Network:
    """A neural network with:
    - n inputs
    - h hidden layers
    - y outputs"""

    def __init__(self, depth, width):
        self.depth = depth
        self.width = width

    def feedforwad(self, x):
        inputs = x.shape[1]
        d = self.depth
        q = self.width
        weights = [np.random.normal() for i in range((inputs * d * q) + q)]

        # need to add layers now that the Layers class is made and working
