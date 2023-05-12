import numpy as np


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        """
        x: numpy array
        """
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            out = x * self.mask
        else:
            out = x * (1.0 - self.dropout_ratio)

        return out

    def backward(self, dout):
        """
        dout: numpy array
        """
        dx = dout * self.mask
        return dx
