import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] # 형상의 처음. 행 갯수 세기
    return -np.sum(t * np.log(y)) / batch_size

def softmax(x):
    c = np.max(x, axis=1)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=1)
    y = exp_x / sum_exp_x

    return y

def mean_squared_error(y, t):
    return (0.5 * np.sum((y - t) ** 2, axis=0)) / y.shape[0]