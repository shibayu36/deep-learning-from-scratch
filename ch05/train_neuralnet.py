import append_path
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# trainの流れ
# 1. ミニバッチの取得
# 2. 勾配の計算
# 3. パラメータの更新
# 4. 1-3を繰り返す

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# hyper parameter
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

# train check data
train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# train
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]  # (100, 784)
    t_batch = t_train[batch_mask]  # (100, 10)

    # Calculate gradient
    grad = network.gradient(x_batch, t_batch)

    # Update parameters
    for key in network.params.keys():
        network.params[key] -= learning_rate * grad[key]

    # Record learning process
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # Calculate accuracy per epoch
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_loss_list[-1], train_acc, test_acc)
