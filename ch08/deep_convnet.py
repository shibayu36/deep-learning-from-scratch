# coding: utf-8
import append_path
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    """認識率99%以上の高精度なConvNet

    ネットワーク構成は下記の通り
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        conv_param_1={"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_2={"filter_num": 16, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_3={"filter_num": 32, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_4={"filter_num": 32, "filter_size": 3, "pad": 2, "stride": 1},
        conv_param_5={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
        conv_param_6={"filter_num": 64, "filter_size": 3, "pad": 1, "stride": 1},
        hidden_size=50,
        output_size=10,
    ):
        # 重みの初期化===========
        # 各層のニューロンひとつあたりが、前層のニューロンといくつのつながりがあるか（TODO:自動で計算する）
        pre_node_nums = np.array(
            [
                1 * 3 * 3,
                16 * 3 * 3,
                16 * 3 * 3,
                32 * 3 * 3,
                32 * 3 * 3,
                64 * 3 * 3,
                64 * 4 * 4,
                hidden_size,
            ]
        )
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLUを使う場合に推奨される初期値

        # <- params: W1 (16, 1, 3, 3), b1 (16,)
        # <- params: W2 (16, 16, 3, 3), b2 (16,)
        # <- params: W3 (32, 16, 3, 3), b3 (32,)
        # <- params: W4 (32, 32, 3, 3), b4 (32,)
        # <- params: W5 (64, 32, 3, 3), b5 (64,)
        # <- params: W6 (64, 64, 3, 3), b6 (64,)
        # <- params: W7 (1024, 50), b7 (50,)
        # <- params: W8 (50, 10), b8 (10,)
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate(
            [
                conv_param_1,
                conv_param_2,
                conv_param_3,
                conv_param_4,
                conv_param_5,
                conv_param_6,
            ]
        ):
            self.params["W" + str(idx + 1)] = weight_init_scales[idx] * np.random.randn(
                conv_param["filter_num"],
                pre_channel_num,
                conv_param["filter_size"],
                conv_param["filter_size"],
            )
            self.params["b" + str(idx + 1)] = np.zeros(conv_param["filter_num"])
            pre_channel_num = conv_param["filter_num"]
        self.params["W7"] = weight_init_scales[6] * np.random.randn(
            64 * 4 * 4, hidden_size
        )
        self.params["b7"] = np.zeros(hidden_size)
        self.params["W8"] = weight_init_scales[7] * np.random.randn(
            hidden_size, output_size
        )
        self.params["b8"] = np.zeros(output_size)

        # Print parameter info
        for key, val in self.params.items():
            print(key, val.shape)
            # print(key, val.size)

        print("sum", sum([v.size for v in self.params.values()]))

        # レイヤの生成===========
        self.layers = []
        self.layers.append(
            Convolution(
                self.params["W1"],
                self.params["b1"],
                conv_param_1["stride"],
                conv_param_1["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(
            Convolution(
                self.params["W2"],
                self.params["b2"],
                conv_param_2["stride"],
                conv_param_2["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(
            Convolution(
                self.params["W3"],
                self.params["b3"],
                conv_param_3["stride"],
                conv_param_3["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(
            Convolution(
                self.params["W4"],
                self.params["b4"],
                conv_param_4["stride"],
                conv_param_4["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(
            Convolution(
                self.params["W5"],
                self.params["b5"],
                conv_param_5["stride"],
                conv_param_5["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(
            Convolution(
                self.params["W6"],
                self.params["b6"],
                conv_param_6["stride"],
                conv_param_6["pad"],
            )
        )
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params["W7"], self.params["b7"]))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params["W8"], self.params["b8"]))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        # x's change of shape
        # input (100, 1, 28, 28)
        # <- params: W1 (16, 1, 3, 3), b1 (16,), pad=1, stride=1
        # Convolution (100, 16, 28, 28)
        # Relu (100, 16, 28, 28)
        # <- params: W2 (16, 16, 3, 3), b2 (16,), pad=1, stride=1
        # Convolution (100, 16, 28, 28)
        # Relu (100, 16, 28, 28)
        # Pooling (100, 16, 14, 14)
        # <- params: W3 (32, 16, 3, 3), b3 (32,), pad=1, stride=1
        # Convolution (100, 32, 14, 14)
        # Relu (100, 32, 14, 14)
        # <- params: W4 (32, 32, 3, 3), b4 (32,), pad=2, stride=1
        # Convolution (100, 32, 16, 16)
        # Relu (100, 32, 16, 16)
        # Pooling (100, 32, 8, 8)
        # <- params: W5 (64, 32, 3, 3), b5 (64,), pad=1, stride=1
        # Convolution (100, 64, 8, 8)
        # Relu (100, 64, 8, 8)
        # <- params: W6 (64, 64, 3, 3), b6 (64,), pad=1, stride=1
        # Convolution (100, 64, 8, 8)
        # Relu (100, 64, 8, 8)
        # Pooling (100, 64, 4, 4)
        # <- params: W7 (1024, 50), b7 (50,)
        # Affine (100, 50)
        # Relu (100, 50)
        # Dropout (100, 50)
        # <- params: W8 (50, 10), b8 (10,)
        # Affine (100, 10)
        # Dropout (100, 10)
        shapes = [["input", x.shape]]
        print("input", x.shape)
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
            print(layer.__class__.__name__, x.shape)
            # shapes.append([layer.__class__.__name__, x.shape])

        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size : (i + 1) * batch_size]
            tt = t[i * batch_size : (i + 1) * batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads["W" + str(i + 1)] = self.layers[layer_idx].dW
            grads["b" + str(i + 1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, "wb") as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, "rb") as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params["W" + str(i + 1)]
            self.layers[layer_idx].b = self.params["b" + str(i + 1)]
