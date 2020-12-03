from node import Node
import numpy as np
import operators


class MLPRegressor:
    def __init__(self, hidden_size:list, activate="relu"):
        self.hidden_size = hidden_size
        self.activate = activate
        self.weight = []
        self.bias = []

    def fit(self, x:np.ndarray, y:np.ndarray):
        x = Node(x)
        y = Node(y)
        self.hidden_size.insert(0, x.tensor.shape[-1])
        for i in range(len(self.hidden_size)):
            if i == len(self.hidden_size)-1:
                w = np.random.uniform(-1, 1, (self.hidden_size[i], 1))
                self.weight.append(Node(w, requires_grad=True))
                b = np.random.uniform(-1, 1, (1,1))
                self.bias.append(Node(b, requires_grad=True))
            else:
                w = np.random.uniform(-1, 1, (self.hidden_size[i], self.hidden_size[i+1]))
                self.weight.append(Node(w, requires_grad=True))
                b = np.random.uniform(-1, 1, (1, self.hidden_size[i+1]))
                self.bias.append(Node(b, requires_grad=True))

        for epoch in range(1000):
            for i in range(len(self.hidden_size)):
                if i == 0:
                    hid = x * self.weight[i] + Node(np.ones((x.tensor.shape[0], 1))) * self.bias[i]
                    hid = hid.relu()
                elif i == len(self.hidden_size) - 1:
                    out = hid * self.weight[i] + Node(np.ones((hid.tensor.shape[0], 1))) * self.bias[i]
                else:
                    hid = hid * self.weight[i] + Node(np.ones((hid.tensor.shape[0], 1))) * self.bias[i]
                    hid = hid.relu()

            loss = (y - out).T() * (y - out)
            loss.backward()
            print("Epoch: {}, loss: {:.4}".format(epoch, loss.tensor[0][0]))

            for w in self.weight:
                w.tensor = w.tensor - 0.001 * w.grad
                w.zeros_grad()

            for b in self.bias:
                b.tensor = b.tensor - 0.001 * b.grad
                b.zeros_grad()
        return

    def predict(self, x:np.ndarray):
        x = Node(x)
        for i in range(len(self.hidden_size)):
            if i == 0:
                hid = x * self.weight[i] + Node(np.ones((x.tensor.shape[0], 1))) * self.bias[i]
                hid = hid.relu()
            elif i == len(self.hidden_size) - 1:
                out = hid * self.weight[i] + Node(np.ones((hid.tensor.shape[0], 1))) * self.bias[i]
            else:
                hid = hid * self.weight[i] + Node(np.ones((hid.tensor.shape[0], 1))) * self.bias[i]
                hid = hid.relu()
        return out

class RNNRegressor:
    def __init__(self, emb_len, seq_len, out_len):
        self.seq_len = seq_len
        self.emb_len = emb_len
        self.out_len = out_len
        self.U = Node(np.random.uniform(-1, 1, (emb_len, emb_len)), requires_grad=True)
        self.W = Node(np.random.uniform(-1, 1, (emb_len, emb_len)), requires_grad=True)
        self.V = Node(np.random.uniform(-1, 1, (emb_len, out_len)), requires_grad=True)


    def fit(self, x:np.ndarray, y:np.ndarray, init_hidden:np.ndarray):
        # x的输入：(样本数, 序列长度, 嵌入维度)，, y(样本数, 输出维度)

        for epoch in range(100):
            total_loss = 0
            for i in range(x.shape[0]):
                hidden = Node(init_hidden)
                trg = Node(y[i])
                loss = Node([[0]])
                for j in range(x.shape[1]):
                    inp = Node([x[i][j]])
                    hidden = (inp * self.U + hidden * self.W).relu()
                    out = hidden * self.V
                    loss = loss + (trg - out).T() * (trg - out)

                loss.backward()
                total_loss += loss.tensor[0][0]
                self.U.tensor = self.U.tensor - 0.001 * self.U.grad
                self.W.tensor = self.W.tensor - 0.001 * self.W.grad
                self.V.tensor = self.V.tensor - 0.001 * self.V.grad

                self.U.zeros_grad()
                self.W.zeros_grad()
                self.V.zeros_grad()
            print("Epoch: {}, Loss: {:.4}".format(epoch, total_loss))

    def predict(self, x:np.ndarray, init_hidden:np.ndarray):
        res_list = []
        for i in range(x.shape[0]):
            init_hidden = Node(init_hidden)
            res = None

            for j in range(x.shape[1]):
                inp = Node(x[j])
                init_hidden = (inp * self.U + init_hidden * self.W).relu()
                out = init_hidden * self.V
                res = operators.concat(res, out)

            self.U.zeros_grad()
            self.V.zeros_grad()
            self.W.zeros_grad()
            res_list.append(res.tensor)
        return res_list


if __name__ == "__main__":
    # 测试MLP
    # mlp = MLPRegressor([5, 3], activate="relu")
    # x = np.random.uniform(-3.14, 3.14, (100, 2))
    # y = np.sin(x[:, 0]) + np.cos(x[:, 1])
    # y = y.reshape((y.shape[0], 1))
    #
    # mlp.fit(x, y)
    # res = mlp.predict(x)

    # # 测试RNN
    rnn = RNNRegressor(emb_len=3, seq_len=4, out_len=1)
    x = np.random.uniform(-1, 1, (20, 6, 3))
    # y = (np.dot(x[:,0], x[:, 1]) + np.dot(x[:,2], x[:,3])) / 2
    y = []

    for i in range(x.shape[0]):
        res = np.dot(x[i][0], x[i][1]) + np.dot(x[i][2], x[i][3])
        res = res / 2
        y.append([[res]])

    rnn.fit(x, y, np.random.uniform(-1, 1, (1, 3)))










