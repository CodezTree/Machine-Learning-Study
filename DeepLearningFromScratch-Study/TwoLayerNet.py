import numpy as np

X = np.array([2, 3])
print(X.shape)
W = np.array([[1, 2, 3],
              [4, 5, 6]])
print(W.shape)

XW = np.dot(X, W)
print(XW.shape)

B = np.array([1, 1, 2])
print(B.shape)

Y = np.add(XW, B)
print(Y.shape)

dY = np.ones_like(Y)
print(dY.shape, dY)

dW = np.dot(np.reshape(X, [2, 1]), np.reshape(dY, [1, 3]))
print(dW.shape, dW)