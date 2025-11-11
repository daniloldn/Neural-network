import numpy as np

def mse_loss(y_true: np.array, y_pred:np.array)-> int:
    return ((y_true - y_pred)**2).mean()