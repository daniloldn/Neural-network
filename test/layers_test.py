import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_network.neuron import Layer


layer_1 = Layer(2, 2)
x = np.array([2,4])

print(layer_1.feed_forward(x))