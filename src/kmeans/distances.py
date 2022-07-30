import numpy as np

def euclidian_dist(a: np.ndarray, b: np.ndarray) -> float:
    
    dist = np.sqrt(
        np.sum(
            (a-b)**2
            )
        )
    return dist

def weighted_euclidian_dist(a: np.ndarray, b: np.ndarray, weights: np.ndarray) -> float:

    weighted_sum = np.sum(
        weights.dot((a-b)**2)
        )

    dist = np.sqrt(
        weighted_sum
        )

    return dist
