import numpy as np
from functions import *

class Affine:
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
        dx = np.dot(dout, self.W.T) # 역전파 흐름에 가중치 전치행렬을 dot product -> 이해 안되면 계산 그래프 그려 이해
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0) # 0 보다 작은 인덱스를 마스킹
        out = x.copy()
        out[self.mask] = 0 # 복사한 out에서 0보다 작은 원소 값을 0으로 만듬

        return out

    def backward(self, dout):
        dout[self.mask] = 0

        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None
        self.loss = None

    def forward(self, y, t):
        self.t = t
        self.y = softmax(y)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size

        return dx * dout
