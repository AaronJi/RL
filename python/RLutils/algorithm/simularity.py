import numpy as np

def cosine_simularity(a, b):
    assert a.shape == b.shape
    a = a.astype(float)
    b = b.astype(float)
    sim = np.dot(a, b.T) / (np.linalg.norm(a, 2) * np.linalg.norm(b, 2))
    return sim