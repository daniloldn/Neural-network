import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_network.network import Network

x = np.array([[1, 2, 3], [2, 3, 4]])
y = np.array([4])

net = Network(x, y, 2, 2)
print(net.loss)
print(net.network)
print((net.network[1])[0].neurons)
