import torch
from torch import nn
from torch.nn import functional as F

# 多层感知机
#  nn.Sequential定义多层感知机
#  线性全连接层20->256
#  Relu激活函数
#  全连接层 256->10
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(1, 20)
# print(net(X))


# 自定义块 pytorch任意一个模型都是基于这个 nn.Moudle来建立的

class MLP(nn.Module):
    '''
    继承nn.Module
    def __init__(self)初始化，用于定义层
    def forward(self,X) 前向推理层
    '''
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
# 创建层
MLP_temp = MLP()
# 输入X之后 调用forward前向推理
# print(MLP_temp(X))

# 顺序块
# 定义块时，定义层的结构， 以list形式传入
# init初始化时 pytorch默认识别self._modules作为这个结构的类
# 前向传播时，循环遍历这个self._modules，调取层出来
class mymodule(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

shunxu_Module = mymodule(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(shunxu_Module(X))

# 在前向传播的时候插入其他操作执行代码
# 自定义操作 在前向操作时可以在任意位置插入不同计算过程
# 以下例子 在relu之前就是X与随机权重矩阵相乘等
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super(FixedHiddenMLP, self).__init__()
        self.rand_weight = torch.rand((20,20), requires_grad=False) # 传入一个随机值，但参与梯度计算（requires_grad=False）

        self.linear = nn.Linear(20,20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1) # torch.mm矩阵乘法
        X = self.linear(X)
        while X.abs().sum()>1:
            X /=2
        return X.sum()

Fixed = FixedHiddenMLP()
# print(Fixed(X))

# 混合搭配各种组合块的方法（灵活）
# 嵌套一个NestMLP linear和上面创建好的FixedHiddenMLP层
class NsetMLP(nn.Module):
    def __init__(self):
        super(NsetMLP, self).__init__()
        self.net = nn.Sequential(nn.Linear(20,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))
chimera = nn.Sequential(NsetMLP(), nn.Linear(16,20), FixedHiddenMLP())
# print(chimera(X))