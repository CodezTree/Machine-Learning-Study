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


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.

    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

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