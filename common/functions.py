import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)  # For overflow
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)


# 二つの確率分布が似ていると値が小さくなるような関数
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    delta = 1e-7  # to handle when y == 0
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + delta)) / batch_size
