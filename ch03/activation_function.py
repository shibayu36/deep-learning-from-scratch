import numpy as np
import matplotlib.pyplot as plt


# x = np.array([-1.0, 1.0, 2.0])
# y = x > 0 # array([]) # => array([False,  True,  True])
# y.astype(np.int32) # => array([0, 1, 1]
def step_function(x):
    y = x > 0
    return y.astype(np.int32)


# x = np.array([-1.0, 1.0, 2.0])
# sigmoid(x)
# array([0.26894142, 0.73105858, 0.88079708])
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
plt.show()
