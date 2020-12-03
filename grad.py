import numpy as np

def grad_add(gradient, lchild, rchild):
    return gradient, gradient

def grad_sub(gradient, lchild, rchild):
    return gradient, -gradient


def grad_mul(gradient, lchild, rchild):
    return np.matmul(gradient, rchild.tensor.T), np.matmul(lchild.tensor.T, gradient)

# 取负号
def grad_neg(gradient, lchild, rchild):
    return -gradient, None


def grad_dot(gradient, lchild, rchild):
    return np.multiply(rchild.tensor, gradient), np.multiply(gradient, lchild.tensor)


def grad_relu(gradient, lchild, rchild):
    tmp_tensor = np.where(lchild.tensor > 0, 1.0, 0.0)
    return np.multiply(gradient, tmp_tensor), None

def grad_T(gradient, lchild, rchild):
    return gradient.T, None



