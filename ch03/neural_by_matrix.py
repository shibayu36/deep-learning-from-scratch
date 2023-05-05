import numpy as np

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])  # [arrows from x1], [arrows from x2]
Y = np.dot(X, W)
print(Y)  # [ 5 11 17 ] = [ y1 y2 y3 ]
