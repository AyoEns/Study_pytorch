import torch
import torch.nn.functional as F
from torch import nn


# 构造一个没有任何参数的自定义层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()

layer = CenteredLayer()
# print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
Y.mean()

# 定义一个带参数的结构层
# 下为定义一个全连接层
class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units)) # nn.Parameter构造一个Parameter和随机初始化，并传入weight中
        self.bias = nn.Parameter(torch.randn(units,))           # 与weight同理
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data # 全连接层 X与weight矩阵相乘 + bias
        return F.relu(linear)

linear = MyLinear(5, 3)
print(linear.weight) # 有梯度
