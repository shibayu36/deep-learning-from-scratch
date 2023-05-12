import numpy as np


class SDG:
    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    # 感覚的には、ボールが動くように、同じ方向に力がかかっているなら加速していくイメージ。
    # これにより、緩やかな傾斜の場合でも、少しずつ大きく進むようになる
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params:
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    """
    パラメータの要素ごとに、学習係数を調整していくモデル
    勾配が大きいほど、より早く学習係数が小さくなる
    """

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params:
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            # 0除算を防ぐために、10e-7を足す
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
