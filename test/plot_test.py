# %%%
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_network.network import Network

# Create dummy data with 100 observations and 5 input features
np.random.seed(42)  # For reproducible results
x = np.random.randn(100, 5)  # 100 observations, 5 features
y = np.random.randn(100, 1)  # 100 target values (single output)

net = Network(x, 3, 5)


train = net.train(x, y, 100, 20, 0.02)


# %%
