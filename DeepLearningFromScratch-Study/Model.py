import numpy as np

from Layers import *
from functions import *
from collections import OrderedDict

class TwoLayerNet: # 기본적인 레이어 두개의 신경망 구현

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):

        # 가중치들을 초기화 시켜준다.
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층을 선언한다
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x) # 데이터를 순전파 시킨다

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : # 목표값의 차원이 1 이 아니다 -> 배치처리
            t = np.argmax(t, axis=1) # 왜 이걸 하는걸까 공부합시다

        accuracy = np.sum(y == t) / float(x.shape[0]) # y == t 의 갯수 값을 입력값의 갯수로 나누어준다

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = dict()
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        # numerical_gradient에 f(function)으로 의미없는 람다로 적용된 self.loss함수를 넘겨줌. 이는 형태를 맞추기 위함
        # numerical_gradient에서의 f를 살려주기 위함. 왜냐하면 f는 x인자를 넘겨주는데 받아주기라도 해야되기 때문
        # 함수가 구현 가능한 이유는 python의 call by reference parameter 특성 때문임
        # 전달받은 loss 함수는 Model의 순전파를 진행시킴

        return grads

    def gradient(self, x, t):
        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout) # 입력까지 쭉 역전파

        grads = dict()
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads