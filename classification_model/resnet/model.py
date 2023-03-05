# bulid resnet

import torch
import torch.nn as nn

class BasciBlock(nn.Module):
    def __init__(self, in_channle, out_channle, stride=1, downsample=None, **kwargs):
        super(BasciBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channle, out_channle, kernel_size=3, stride=stride, padding=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(out_channle)
        self.relu = nn.ReLU()
        self.Conv2 = nn.Conv2d(in_channle, out_channle, kernel_size=3, stride=stride, padding=1, bias=False)
        self.Bn2 = nn.BatchNorm2d(out_channle)
        self.downsample = downsample


    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.Conv1(x)
        out = self.Bn1(out)
        out = self.relu(out)
        out = self.Conv2(out)
        out = self.Bn2(out)
        out += identity
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channle, out_channle, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()
        width = int(out_channle * (width_per_group/64.0)) * groups

        self.Conv1 = nn.Conv2d(in_channle, width, kernel_size=1,stride=1, bias=False)
        self.Bn1 = nn.BatchNorm2d(width)

        self.Conv2 = nn.Conv2d(width, width, kernel_size=3, groups=groups, stride=stride, bias=False, padding=1)
        self.Bn2 = nn.BatchNorm2d(width)

        self.Conv3 = nn.Conv2d(width, out_channle*self.expansion, kernel_size=1, stride=1, bias=False)
        self.Bn3 = nn.BatchNorm2d(out_channle*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.Conv1(x)
        out = self.Bn1(out)
        out = self.relu(out)

        out = self.Conv2(out)
        out = self.Bn2(out)
        out = self.relu(out)

        out = self.Conv3(out)
        out = self.Bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, groups=1, width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group
        self.Conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.Bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channle, blocks_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channle * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channle * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channle * block.expansion)
            )
        layer = []
        layer.append(block(
            self.in_channel,
            channle,
            downsample=downsample,
            stride = stride,
            groups = self.groups,
            width_per_group = self.width_per_group
        ))
        self.in_channel = channle * block.expansion

        for _ in range(1, blocks_num):
            layer.append(block(
                self.in_channel,
                channle,
                groups=self.groups,
                width_per_group=self.width_per_group
            ))

        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.Conv1(x)
        x = self.Bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


if __name__ == '__main__':
    ResNet50 = resnet50()
    print(ResNet50)