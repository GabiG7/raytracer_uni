import numpy as np


def normalize_vector(vector):
    vector /= np.linalg.norm(vector)
    return vector
