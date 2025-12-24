import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w = np.asarray(w, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)
    if 0 <= momentum < 1 and lr > 0:
        wlook = w - momentum*v
        v_new = momentum*v + lr*grad
        w_new = w - v_new
        return w_new, v_new