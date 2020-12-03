import numpy as np
from grad import grad_add, grad_sub, grad_mul, grad_neg, grad_relu, grad_T
import torch

class Node:

    def __init__(self, tensor, requires_grad=False):
        self.tensor = np.array(tensor)
        self.requires_grad = requires_grad
        self.grad = 0.0
        self.grad_fn = None
        self.is_leaf = True
        self.father = None
        self.lchild = None
        self.rchild = None

        self.param = ""

    # 左侧调用的add
    def __add__(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = Node(value)  # requires_grad=False

        n_tensor = self.tensor + value.tensor
        n_node = Node(n_tensor)
        n_node.requires_grad = self.requires_grad or value.requires_grad

        n_node.grad_fn = grad_add
        n_node.is_leaf = False
        n_node.lchild = self
        n_node.lchild.father = n_node
        n_node.rchild = value
        n_node.rchild.father = n_node

        n_node.param = "add"
        return n_node

    def __radd__(self, other):
        return self.__add__(other)  # 这边交换了左右操作数，但计算grad时并不影响

    def __sub__(self, value):
        if isinstance(value, (list, np.ndarray)):
            value = Node(value)  # requires_grad=False

        n_tensor = self.tensor - value.tensor
        n_node = Node(n_tensor)

        n_node.requires_grad = self.requires_grad or value.requires_grad
        n_node.grad_fn = grad_sub
        n_node.is_leaf = False
        n_node.lchild = self
        n_node.lchild.father = n_node
        n_node.rchild = value
        n_node.rchild.father = n_node

        n_node.param = "sub"
        return n_node

    def __rsub__(self, other):
        if isinstance(other, (list, np.ndarray)):
            other = Node(other)

        if isinstance(other, (int, float)):
            other = Node(np.ones(self.tensor.shape, dtype=np.float) * other)

        n_tensor = other.tensor - self.tensor
        n_node = Node(n_tensor)

        n_node.requires_grad = self.requires_grad or other.requires_grad
        n_node.grad_fn = grad_sub
        n_node.is_leaf = False
        n_node.lchild = other
        n_node.lchild.father = n_node
        n_node.rchild = self
        n_node.rchild.father = n_node

        n_node.param = "rsub"
        return n_node

    def __neg__(self):
        n_tensor = -self.tensor
        n_node = Node(n_tensor)

        n_node.requires_grad = self.requires_grad
        n_node.grad_fn = grad_neg
        n_node.is_leaf = False
        n_node.lchild = self
        n_node.lchild.father = n_node
        n_node.rchild = None
        return n_node

    # 乘法需要分矩阵乘法、数值*矩阵、矩阵元素相乘
    def __mul__(self, other):
        if isinstance(other, (list, np.ndarray)):
            other = Node(other)
        elif isinstance(other, (int, float)):
            other = Node(np.ones(self.tensor.shape) * other)

        n_tensor = np.matmul(self.tensor, other.tensor)
        n_node = Node(n_tensor)

        n_node.requires_grad = self.requires_grad or other.requires_grad
        n_node.grad_fn = grad_mul
        n_node.is_leaf = False
        n_node.lchild = self
        n_node.lchild.father = n_node
        n_node.rchild = other
        n_node.rchild.father = n_node

        n_node.param = "mul"
        return n_node

    def __rmul__(self, other):
        if isinstance(other, (list, np.ndarray)):
            other = Node(other)
        elif isinstance(other, (int, float)):
            other = Node(np.ones(self.tensor.shape) * other)

        n_tensor = np.matmul(other.tensor, self.tensor)
        n_node = Node(n_tensor)

        n_node.requires_grad = self.requires_grad or other.requires_grad
        n_node.grad_fn = grad_mul
        n_node.is_leaf = False
        n_node.lchild = other
        n_node.lchild.father = n_node
        n_node.rchild = self
        n_node.rchild.father = n_node

        n_node.param = "rmul"
        return n_node

    def relu(self):
        n_tensor = np.where(self.tensor > 0, self.tensor, 0)
        n_node = Node(n_tensor)

        n_node.requires_grad = self.requires_grad
        n_node.grad_fn = grad_relu
        n_node.is_leaf = False
        n_node.lchild = self
        n_node.lchild.father = n_node
        n_node.rchild = None

        n_node.param = "relu"
        return n_node


    def sum(self, axis=0):
        if axis == 0:
            ones = np.ones((1,self.tensor.shape[0]))
            n_node = self.__rmul__(ones)
        elif axis == 1:
            ones = np.ones((self.tensor.shape[1], 1))
            n_node = self.__mul__(ones)
        return n_node


    def T(self):
        n_tensor = self.tensor.T.copy()
        n_node = Node(n_tensor)

        n_node.requires_grad = self.requires_grad
        n_node.grad_fn = grad_T
        n_node.is_leaf = False
        n_node.lchild = self
        n_node.lchild.father = n_node
        n_node.rchild = None

        n_node.param = "T"
        return n_node



    # 反向传播过程
    def backward(self, gradient=np.ones((1, 1), dtype=np.float)):
        if not self.requires_grad:
            return

        self.grad += gradient
        if not self.is_leaf:
            left_grad, right_grad = self.grad_fn(gradient, self.lchild, self.rchild)
            if self.lchild is not None:
                if self.lchild.requires_grad:
                    self.lchild.backward(gradient=left_grad)

            if self.rchild is not None:
                if self.rchild.requires_grad:
                    self.rchild.backward(gradient=right_grad)
    # 清空梯度
    def zeros_grad(self):
        self.grad = 0
        if self.lchild is not None:
            self.lchild.zeros_grad()
        if self.rchild is not None:
            self.rchild.zeros_grad()
        return



if __name__ == "__main__":
    # 简单测试与pytorch对比
    w1 = np.random.uniform(-1, 1, (2, 5))
    w2 = np.random.uniform(-1, 1, (5, 1))

    b1 = np.random.uniform(-1, 1, (1, 5))
    b2 = np.random.uniform(-1, 1, (1, 1))

    x = np.random.uniform(-3.14, 3.14, (100, 2))
    y = np.sin(x[:, 0]) + np.cos(x[:, 1])
    y = y.reshape((y.shape[0], 1))

    # node 过程
    W1 = Node(w1, requires_grad=True)
    W2 = Node(w2, requires_grad=True)
    B1 = Node(b1, requires_grad=True)
    B2 = Node(b2, requires_grad=True)

    hid = Node(x) * W1 + Node(np.ones((x.shape[0], 1))) * B1
    hid = hid.relu()
    out = hid * W2 + Node(np.ones((hid.tensor.shape[0], 1))) * B2
    loss = (Node(y) - out).T() * (Node(y) - out)
    loss.backward()

    # tensor 过程
    WW1 = torch.tensor(w1, requires_grad=True)
    WW2 = torch.tensor(w2, requires_grad=True)
    BB1 = torch.tensor(b1, requires_grad=True)
    BB2 = torch.tensor(b2, requires_grad=True)

    hhid = torch.mm(torch.tensor(x), WW1) + torch.mm(torch.tensor(np.ones((x.shape[0], 1))), BB1)
    hhid = hhid.relu()
    oout = torch.mm(hhid, WW2) + torch.mm(torch.tensor(np.ones((hhid.shape[0], 1))), BB2)
    lloss = torch.mm((torch.tensor(y) - oout).T , (torch.tensor(y) - oout))
    lloss.backward()

    print("Pytorch W1: ")
    print("{}".format(" ".join(str(e.data.float())for e in WW1[0])))
    print("{}".format(" ".join(str(e.data.float()) for e in WW1[1])))

    print("Node W1: ")
    print("{}".format(" ".join(str(e) for e in W1.tensor[0])))
    print("{}".format(" ".join(str(e) for e in W1.tensor[1])))

    print("Pytorch W2: ")
    print("{}".format(" ".join(str(e) for e in WW2)))

    print("Node W2: ")
    print("{}".format(" ".join(str(e) for e in W2.tensor)))

    print("Pytorch B1: ")
    print("{}".format(" ".join(str(e) for e in BB1)))

    print("Node B1: ")
    print("{}".format(" ".join(str(e) for e in B1.tensor)))

    print("Pytorch B2: ")
    print("{}".format(" ".join(str(e) for e in BB2)))

    print("Node B2: ")
    print("{}".format(" ".join(str(e) for e in B2.tensor)))





# 简单的测试
# if __name__ == "__main__":
#
#     W1 = Node(np.random.uniform(-1, 1, (2, 5)), requires_grad=True)
#     W2 = Node(np.random.uniform(-1, 1, (5, 1)), requires_grad=True)
#     B1 = Node(np.random.uniform(-1, 1, (1, 5)), requires_grad=True)
#     B2 = Node(np.random.uniform(-1, 1, (1, 1)), requires_grad=True)
#
#     x = np.random.uniform(-3.14, 3.14, (100, 2))
#     y = np.sin(x[:, 0]) + np.cos(x[:, 1])
#     y = y.reshape((y.shape[0], 1))
#
#     for epoch in range(1000):
#
#         print("Epoch : {}".format(epoch))
#
#         hid = Node(x) * W1 + Node(np.ones((x.shape[0], 1))) * B1
#         hid = hid.relu()
#         out = hid * W2 + Node(np.ones((hid.tensor.shape[0], 1))) * B2
#         loss = (Node(y) - out).T() * (Node(y) - out)
#
#         loss.backward()
#         W1.tensor = W1.tensor - 0.001 * W1.grad
#         W2.tensor = W2.tensor - 0.001 * W2.grad
#         B1.tensor = B1.tensor - 0.001 * B1.grad
#         B2.tensor = B2.tensor - 0.001 * B2.grad
#
#         W1.zeros_grad()
#         W2.zeros_grad()
#         B1.zeros_grad()
#         B2.zeros_grad()
#
#         print("计算RMSE")
#         total_loss = 0.0
#
#         for i in range(x.shape[0]):
#             xx = Node([x[i]])
#             yy = Node([[y[i][0]]])
#             hid = xx * W1 + Node(np.ones((xx.tensor.shape[0], 1)))*B1
#             hid = hid.relu()
#             out = hid * W2 + Node(np.ones((hid.tensor.shape[0], 1)))*B2
#             loss = (yy - out) * (yy - out)
#             total_loss += loss.tensor[0][0]
#             W1.zeros_grad()
#             W2.zeros_grad()
#             B1.zeros_grad()
#             B2.zeros_grad()
#
#         total_loss = total_loss / x.shape[0]
#         total_loss = np.sqrt(total_loss)
#         print("{}".format(total_loss))

