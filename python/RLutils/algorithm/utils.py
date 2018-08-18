
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

# x is a vector
def softmax_power(x, power=1.0):
    """Compute the softmax with sharper options."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    exp_x_power = exp_x**power
    softmax_x = exp_x_power / np.sum(exp_x_power)

    return softmax_x

# x is a vector; scale to [-1, 1]
def minMaxScale(x):
    small = 1.0e-5

    x_min = np.min(x)
    x_max = np.max(x)

    if x_max - x_min < small:
        return x

    x_scale = (2*x - (x_max + x_min))/(x_max - x_min)

    return x_scale

def scaler(theta, scale_way):
    if scale_way == "sigmoid":
        return sigmoid(theta)
    if scale_way == "minMax":
        return minMaxScale(theta)

    return theta

def get_dict_subset(d, sub_key_list):
    return {k: v for k, v in d.items() if k in sub_key_list}

def sort_dict_by_value(d, reverse=True):
    #d = sorted(d.items(), key=lambda x: x[1], reverse=reverse)

    import operator
    d = sorted(d.items(), key=operator.itemgetter(1), reverse=reverse)
    return d

def list_find(l, element):
    for i, e in enumerate(l):
        if e == element:
            return i
    return None
