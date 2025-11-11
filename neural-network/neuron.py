import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def feed_forward(self, inputs):
        total = np.dot(self.weight, inputs) + self.bias
        return sigmoid(total)




