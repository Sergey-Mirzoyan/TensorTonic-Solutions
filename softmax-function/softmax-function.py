import numpy as np

def softmax(x):
    x = np.asarray(x, dtype=float)
    max_x = np.max(x, axis=-1, keepdims=True)  # Максимум по последней оси
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)