import append_path

from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
print(x_train.shape)
print(x_train)
print(t_train.shape)
print(t_train)
print(x_test.shape)
print(t_test.shape)
