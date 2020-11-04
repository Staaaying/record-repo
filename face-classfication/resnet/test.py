import os
from matplotlib import image
import numpy as np
import torch as t
import torch.nn as nn
import matplotlib.image as image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class MyDataSet(Dataset):
    '''
    定义数据集，用于将读取到的图片数据转换并处理成CNN神经网络需要的格式
    '''
    def __init__(self, DataArray, LabelArray):
        super(MyDataSet, self).__init__()
        self.data = DataArray
        self.label = LabelArray

    def __getitem__(self, index):
        # 对图片的预处理步骤
        # 1. 中心缩放至224(ResNet的输入大小)
        # 2. 随机旋转0-30°
        # 3. 对图片进行归一化，参数来源为pytorch官方文档
        im_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(size=224),
            transforms.RandomRotation((0, 30)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return im_trans(self.data[index]), t.tensor(self.label[index], dtype=t.long)

    def __len__(self):
        return self.label.shape[0]

# 读取LFW数据集，将图片数据读入数组并将名字转换为标签
path = r'face+\lfw'
pathlist = map(lambda x: '\\'.join([path, x]), os.listdir(path))
namedict = {}
data, label, label_xx = [], [], []
idx = 0
for item in pathlist:
    dirlist = os.listdir(item)
    if not (30<= len(dirlist) <= 100):
        continue
    namedict[str(idx)] = item.split('\\')[-1]
    idx += 1
path = r'face+\lfw_test'
pathlist = map(lambda x: '\\'.join([path, x]), os.listdir(path))
for item in pathlist:
    data.append(image.imread(item))
    label.append(item.split('\\')[-1][:-9])
    label_xx.append(1)

data, label_xx = np.stack(data), np.array(label_xx)
TestSet = MyDataSet(data, label_xx)
resnet = t.load(r'face+\resnet.pth')

# 打印测试集的真实/预测结果
for idx, (data, _) in enumerate(TestSet):
    Outputs = resnet(data.unsqueeze(0))
    _, pred = t.max(Outputs.data, 1)
    pred_name = namedict[str(pred.item())]
    real_name = label[idx]
    print('real name: {}\t\t\t\t\tpredict name: {}'.format(real_name, pred_name))
