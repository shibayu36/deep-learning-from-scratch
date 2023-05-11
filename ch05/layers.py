import numpy as np
from common.functions import cross_entropy_error, sigmoid, softmax


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        """
        x: numpy array
        """
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        """
        dout: numpy array
        """
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        """
        x: numpy array
        """
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        """
        dout: numpy array
        """
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine:
    """
    Affine transformation layer for batch input
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # loss
        self.y = None  # output of softmax
        self.t = None  # teacher data (one-hot vector)

    def forward(self, x, t):
        # x's shape is (batch_size, 10)
        # t's shape is (batch_size, 10)
        # y's shape is (batch_size, 10)
        self.t = t
        self.y = softmax(x)
        # loss shape is (batch_size, 1)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]  # number of data in a batch
        # q: why do we divide by batch_size?
        # a: because we want to calculate the average of loss
        dx = (self.y - self.t) / batch_size
        # dx's shape is (batch_size, 10)
        return dx
