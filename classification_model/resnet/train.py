import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import resnet50

# train Alex

def main():
    '''训练用的函数'''
    # 调用当前gpu设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} devices.".format(device))

    # torchvision.transforms是pytorch中的图像预处理包。
    # RandomResizedCrop 将PIL图像裁剪成任意大小和纵横比
    # RandomHorizontalFlip 以0.5的概率水平翻转给定的PIL图像
    # ToTensor 转换到Tensor格式
    # Normalize 用均值和标准差归一化张量图像
    data_transform = {
        "train": transforms.Compose(
            [transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
        ),

        "val": transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "stduy_mtorch\\data_set", "flower_data")
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    # ImageFolder是一个通用的数据加载器

    # root：图片存储的根目录，即各类别文件夹所在目录的上一级目录。
    # transform：对图片进行预处理的操作（函数），原始图片作为输入，返回一个转换后的图片。
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)

    # ImageFolder函数假设所有的文件按文件夹保存，每个文件夹下存储同一类别的图片，相当于类别
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    # 批处理数量
    batch_size = 16
    # nw 是否开启多线程 0就是默认
    nw = min([os.cpu_count(), batch_size if batch_size>1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    # 数据加载器，结合了数据集和取样器，并且可以提供多个线程处理数据集。
    # 在训练模型时使用到此函数，用来 把训练数据分成多个小组 ，此函数 每次抛出一组数据 。直至把所有的数据都抛出。
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform['val'])
    val_num = len(validate_dataset)

    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=4, shuffle=False, num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # 创建模型 ResNet
    net = resnet50(num_classes=5, include_top=True)
    # 模型传入到cuda即GPU中进行训练
    net.to(device)
    # CE loss
    loss_function = nn.CrossEntropyLoss()
    # Adam优化器
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    # 训练步骤
    epochs = 5
    save_path = "./Resnet50.pth"
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        # 开始训练
        net.train()
        running_loss = 0.0
        # tqdm这个包来显示训练的进度
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            # 选数据和lable
            image, label = data

            optimizer.zero_grad()                           # 梯度初始化为零，把loss关于weight的导数变成0
            outputs = net(image.to(device))                 # forward：将数据传入模型，前向传播求出预测的值
            loss = loss_function(outputs, label.to(device)) # 求loss
            loss.backward()                                 # 反向传播
            optimizer.step()                                # 更新所有参数

            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        # model.eval()，不启用 BatchNormalization 和 Dropout。此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                # 同样取data
                val_image, val_lable = val_data
                outputs = net(val_image.to(device))         # 求验证值
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_lable.to(device)).sum().item()

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print("train Done!")

if __name__ == '__main__':
    main()