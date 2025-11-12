import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_network.neuron import Neuron

n1 = Neuron(2, 1)
x = np.array([1, 2])
print(n1.id, n1.weights, n1.bias)
print(n1.feed_forward(x))
