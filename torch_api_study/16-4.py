import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file') # 写文件

x2 = torch.load("x-file") # 读文件
# print(x2)

# 保存权重文件和加载权重文件
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)
torch.save(net.state_dict(), 'mlp.params') # net.state_dict() 拿出数据 并保存

mlp_params = torch.load('mlp.params')
print(mlp_params)

clone = MLP()
clone.load_state_dict(torch.load('mlp.params')) # 定义一个相同的网络结构 内置函数load_state_dict 加载权重
clone.eval()
Y_clone = clone(X)
print(Y_clone == Y)