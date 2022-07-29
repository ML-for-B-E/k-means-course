import numpy as np

def euclidian_dist(a: np.ndarray, b: np.ndarray) -> float:
    
    dist = np.sqrt(
        np.sum(
            (a-b)**2
            )
        )
    return dist