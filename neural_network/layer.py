from .neuron import Neuron

class Layer:
    """a layer with n neurons"""
    
    def __init__(self, n_inputs: int, n_neurons: int, activation: Optional[Callable] = None):
        self.neurons = [Neuron(n_inputs, i, activation) for i in range(n_neurons)]

    def feed_forward(self, inputs):
        self.output = [self.neurons[i].feed_forward(inputs) for i in range(len(self.neurons))]
        return self.output