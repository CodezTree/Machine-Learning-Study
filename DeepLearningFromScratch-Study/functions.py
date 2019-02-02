import numpy as np

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] # 형상의 처음. 행 갯수 세기
    return -np.sum(t * np.log(y)) / batch_size

def softmax(x):
    c = np.max(x, axis=1, keepdims=x.ndim)
    exp_x = np.exp(x - c)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=x.ndim)
    y = exp_x / sum_exp_x

    return y

def mean_squared_error(y, t):
    return (0.5 * np.sum((y - t) ** 2, axis=0)) / y.shape[0]

def numerical_gradient(f, x):
    """
    수치적 미분 구현 함수

    ------------

    f : Gradient를 구할 함수
    x : 입력 값

    ------------
    """

    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 입력 값과 같은 형상을 취한다 (여기선 가중치 행렬이 입력이 될 수도 있다

    # for idx in range(x.size): # 입력 배열 혹은 행렬의 모든 원소 하나씩 접근 ( Gradient 를 구하기 위해 )
    #     tmp_val = x[idx] # 현재 계산하고자 하는 원소 값을 임시로 저장
    #
    #     # 해당 원소 부분만 국소적 미분을 취한다 ( 수치 편미분 )
    #     x[idx] = tmp_val + h # 우극한
    #     fxh1 = f(x)
    #
    #     x[idx] = tmp_val - h # 좌극한
    #     fxh2 = f(x)
    #
    #     grad[idx] = (fxh1 - fxh2) / (2 * h) # 편미분계수를 구함
    #     x[idx] = tmp_val # 계산 후 다른 원소의 편미분 위해 원래대로 돌려놓아 준다.

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite']) # 다차원 배열 처리를 위해 iterator로 만들어줌
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

return grad


### PRACTICE

# y = np.array([[0.3, 0.2, 0.1, 0.3, 1],
#               [0.2, 0.15, 0.8, 0.11, 0.25],
#               [0.13, 2, 0.1, 0.07, 0.12]])
#
# y = y * 2
# x
# t = np.array([[0, 0, 0, 0, 1],
#               [0, 0, 1, 0, 0],
#               [0, 1, 0, 0, 0]])
# z = softmax(y)
# print(z)
# print(np.sum(z, axis=1))