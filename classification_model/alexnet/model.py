# AlexNet Model

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    '''定义AlexNet模型结构'''
    def __init__(self, num_classes=1000, init_weight=False):
        '''包括主干模型和分类模型'''
        super(AlexNet, self).__init__()
        self.alex_model = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,128,kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(128*6*6,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048,2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weight==True:
            self._initialize_weights()

    def _initialize_weights(self):
        '''初始化，对卷积层进行hekaiming初始化，全连接层进行普通初始化'''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0,0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        '''前向传播'''
        x = self.alex_model(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    alex_mode = AlexNet()
    print(alex_mode)