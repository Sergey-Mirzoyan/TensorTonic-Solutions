import numpy as np

def relu(x):
    x = np.asarray(x, dtype=float)
    out = np.maximum(0.0, x)
    if out.ndim == 0:  # если скаляр
        return np.array([out])
    return out