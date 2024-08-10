import numpy as np


def normalize_vector(vector):
    norm = np.linalg.norm(vector)

    # Case of norm is zero
    if norm == 0:
        return vector

    # Ensure value type
    vector = vector.astype(np.float64)

    vector /= norm
    return vector
