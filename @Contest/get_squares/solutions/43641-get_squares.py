import numpy as np

def get_squares(a: np.ndarray) -> float:
    return (a[a > 0] ** 2).sum()

