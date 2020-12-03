import numpy as np
import torch
from node import Node

def concat(A: Node, B: Node, axis=1):               # 这里先做1维的concat
    '''
    :param A: A矩阵
    :param B: B矩阵
    :return: 拼接后的[A,B]矩阵，因为底层是矩阵，那么
    拼接函数可以这样实现：Y = A * [I|0] + B * [0|I]
    '''

    if B is None:                                   # 默认
        return A
    if A is None:
        return B

    mask1 = np.zeros((A.tensor.shape[-1], A.tensor[-1]+B.tensor.shape[-1]))
    mask2 = np.zeros((B.tensor.shape[-1], B.tensor.shape[-1]+A.tensor.shape[-1]))

    for i in range(A.tensor.shape[-1]):
        mask1[i][i] = 1
    for j in range(B.tensor.shape[-1]):
        mask2[j+A.tensor.shape[-1]][j] = 1
    mask1 = Node(mask1)
    mask2 = Node(mask2)

    Y = A * mask1 + B * mask2
    return Y







