import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet50


def main():
    # 预测
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据类型转换格式 PIL转tensor
    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    # 图片路径 打开图片
    img_path = "D:\\Pycharm_project\\stduy_mtorch\data_set\\flower_data\\flower_photos\\tulips\\11746276_de3dec8201.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)
    # [N, C, H, W]
    # 图像转换
    img = data_transform(img)
    # expand batch dimension
    # 增加batch_size纬度
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = resnet50(num_classes=5).to(device)

    # load model weights
    weights_path = "./resnet50.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    # 模型权重加载
    model.load_state_dict(torch.load(weights_path))

    # 模型越策
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    # 预测结果
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()
