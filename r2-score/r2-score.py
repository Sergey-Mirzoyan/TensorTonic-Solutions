import numpy as np

def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    r2 = None  # или инициализируем значением по умолчанию
    
    if np.all(y_true == y_true[0]):
        if np.all(y_true == y_pred):
            r2 = 1.0
        else:
            r2 = 0.0
    else:
        sse = np.sum((y_true - y_pred) ** 2)
        sst = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - (sse / sst)
    
    return float(r2)