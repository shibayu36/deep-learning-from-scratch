import append_path
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)
x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
idx = np.argmax(p)
print(idx)
t = np.array([0, 0, 0])
t[idx] = 1
print(net.loss(x, np.array([1, 0, 0])))
print(net.loss(x, np.array([0, 1, 0])))
print(net.loss(x, np.array([0, 0, 1])))


def f(W):
    return net.loss(x, t)


dW = numerical_gradient(f, net.W)
print(dW)
