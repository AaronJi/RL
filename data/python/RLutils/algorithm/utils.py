
import numpy as np

# x can be a vector
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# x is a vector
def softmax(x):
    """Compute the softmax in a numerically stable way."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)

    return softmax_x
