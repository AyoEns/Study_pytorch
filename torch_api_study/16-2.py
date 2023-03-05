import torch
from torch import nn
# 参数管理
net = nn.Sequential(nn.Linear(4,8),nn.ReLU(), nn.Linear(8,1))
X = torch.rand(size=(2,4))
# print(net(X))
# print(net[2].state_dict()) # 看成是一个list net[2]为 nn.Linear(8,1)的数据
# print(type(net[2].bias)) # <class 'torch.nn.parameter.Parameter'>
# print(net[2].bias.data) # 通过data访问tensor数值
# print(net[2].weight.grad == None) # weight.grad访问梯度  因为此处没有反向传播 所以梯度==None
# print(*[(name, param.shape) for name, param in net.named_parameters()]) # 取出网络各层的名字和shaoe
# print(net.state_dict()['2.bias'].data) # 根据字典中名字访问对应的数据

# 有嵌套时的访问参数
def block1():
    return nn.Sequential(nn.Linear(4,8), nn.ReLU(), nn.Linear(8,4), nn.ReLU())

def block2():
    net = nn.Sequential()
    for i in range(4):
        # add_module 模块名字，加模块层（函数）
        net.add_module(f"block{i}",block1())
    return net

rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
X = rgnet(X)
print(rgnet) # 打印网络数据
# print(*[(name, param.shape) for name, param in rgnet.named_parameters()]) # 取出网络各层的名字和shaoe

# 内置初始化参数
# 也可以对不同块进行初始块，
def init_nomrmal(m):
    '''对m层为全连接层的weight初始化为均值为0，方差为0.1，bias置0的初始化操作'''
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.1) # normal_ 方法 初始化
        nn.init.zeros_(m.bias)

# net.apply(init_nomrmal) # apply遍历整个网络结构，进行初始化
# print(net[0].weight.data[0], net[0].bias.data[0]) # 查看初始化结果 tensor([-0.0183,  0.1059,  0.1649,  0.1141]) tensor(0.)

def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1) # constant_ 方法 初始化为1
        nn.init.zeros_(m.bias)

# net.apply(init_constant) # apply遍历整个网络结构，进行初始化为1
# print(net[0].weight.data[0], net[0].bias.data[0]) # 查看初始化结果 tensor([1., 1., 1., 1.]) tensor(0.)

def xvaier(m):
    '''xvaier初始化'''
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# 分别对不同块进行初始化
net[0].apply(xvaier)
net[2].apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0]) # tensor([0.6808, 0.2293, 0.4682, 0.0941]) tensor(0.2833)
print(net[2].weight.data[0], net[2].bias.data[0]) # tensor([1., 1., 1., 1., 1., 1., 1., 1.]) tensor(0.)

# 参数绑定
shared = nn.Linear(8,8)
# shared同一内存 ，所以无论怎么更新 net1内2个shard的权重值都是相同的
net1 = nn.Sequential(nn.Linear(4,8),nn.ReLU(), shared, nn.ReLU(), shared, nn.ReLU(), nn.Linear(8,1))
# net1(X)
print(net1[2].weight.data[0] == net1[4].weight.data[0])
net1[2].weight.data[0,0] = 100
print(net1[2].weight.data[0] == net1[4].weight.data[0])