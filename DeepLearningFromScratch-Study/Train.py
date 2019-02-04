import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from Model import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size = 784, hidden_size = 80, output_size = 10)

train_size = x_train.shape[0]
iters_num = 10000 # 10000번 학습을 진행한다
batch_size = 100
learning_rate = 0.1

iters_per_epoch = max(train_size / batch_size, 1) # 학습 정보를 나타내 줄 횟수를 구한다 (?)

for i in range(iters_num):
    # 학습을 시킬 데이터를 무작위로 추출한다
    train_mask = np.random.choice(train_size, batch_size) # train 가능 사진 넘버 range에서 batch개 만큼 뽑아서 리스트

    X_train, T_train = x_train[train_mask], t_train[train_mask] # 데이터 셋에서 선정된 원소만 불러온다
    grads = network.gradient(X_train, T_train) # Feed Forwarding 동시에 gradient를 구해 불러온다.

    # loss = network.loss(X_train, T_train) # 신경망의 loss를 구한다
    # print('loss : '+str(loss))

    # 오차역전파 with Delta Rule
    for key in grads.keys():
        network.params[key] -= learning_rate * grads[key] # 해당하는 가중치를 갱신한다

    if i % iters_per_epoch == 0: # 데이터 표시 횟수 도달 -> 전체 데이터에 대해서
        train_acc = network.accuracy(x_train, t_train) # 전체 학습 데이터에 대해 정확도 계산
        test_acc = network.accuracy(x_test, t_test) # 전체 테스트 데이터에 대해 정확도 계산
        print('train_acc, test_acc : ' + str(train_acc) + ' , ' + str(test_acc))

