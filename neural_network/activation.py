import numpy as np

def sigmoid(x: float)-> float:
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x:float)-> float:
    fx = sigmoid(x)
    return fx * (1-fx)