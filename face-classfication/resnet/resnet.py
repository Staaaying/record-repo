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
data, label = [], []
idx = 0
for item in pathlist:
    dirlist = os.listdir(item)
    # 选取拥有30-100张照片的人作为数据来源
    # 太少网络不容易学习到其人脸特征，太多的话则容易过拟合
    if not (30<= len(dirlist) <= 100):
        continue
    # data:     存储人像照片的三通道数据
    # label:    存储人像的对应标签(整数)
    # namedict: 记录label中整数与人名的对应关系
    for picpath in dirlist:
        data.append(image.imread(item + '\\' + picpath))
        label.append(idx)
    namedict[str(idx)] = item.split('\\')[-1]
    idx += 1

# 随机打乱数据，重新排序并按照8:2的比例分割训练集和测试集
data, label = np.stack(data), np.array(label)
idx = np.random.permutation(data.shape[0])
data, label = data[idx], label[idx]
train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2)

# 将分割好的训练集和测试集处理为pytorch所需的格式
TrainSet = MyDataSet(train_X, train_Y)
TestSet = MyDataSet(test_X, test_Y)
TrainLoader = DataLoader(TrainSet, batch_size=32, shuffle=True, drop_last=True)
TestLoader = DataLoader(TestSet, batch_size=32, shuffle=True, drop_last=True)

# 调用预训练的resnet18进行迁移学习
# resnet50参数量过多，训练效果不太好
resnet = models.resnet18(pretrained=True)
# for param in resnet.parameters():
#     param.requires_grad = False

# 将resnet的输出fc(全连接层)替换为本任务所需的格式
# 1000-->256-->relu-->dropout-->29-->softmax
fc_inputs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(256, 29)
).cuda()

# 定义交叉熵损失函数和Adam优化器(学习率，权重衰减使用默认值)
loss = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(resnet.parameters())

def train(net, dataloader, testdataloader, optimizer, criterion, epocs=20):
    # 以下四个参数分别用于存储训练和测试的损失函数值以及分类准确率
    train_loss_arr, train_acc_arr, test_loss_arr, test_acc_arr = [], [], [], []
    for epoc in range(epocs):
        net.train()
        TrainLoss, TrainAcc = 0, 0
        for BatchIdx, (InputData, Labels) in enumerate(dataloader):
            Outputs = net(InputData)
            optimizer.zero_grad()
            loss = criterion(Outputs.squeeze(), Labels)
            loss.backward()
            optimizer.step()
            TrainLoss += loss.item()
            _, pred = t.max(Outputs.data, 1)
            TrainAcc += t.mean(pred.eq(Labels.data.view_as(pred)).type(t.FloatTensor)).item() * len(InputData)
            if BatchIdx % 10 == 0 and BatchIdx > 0:
                print('Bathch: {}/{}\tLoss: {}\tAcc: {}%'.format(BatchIdx, len(dataloader), round(TrainLoss, 2), 
                                                                 round(100*TrainAcc/((BatchIdx+1) * InputData.shape[0]), 2)))
        train_acc_arr.append(100*TrainAcc/(len(dataloader)*32))
        train_loss_arr.append(TrainLoss)
        TestLoss, TestAcc = 0, 0
        with t.no_grad():
            net.eval()
            for BatchIdx, (InputData, Labels) in enumerate(testdataloader):
                Outputs = net(InputData)
                loss = criterion(Outputs.squeeze(), Labels)
                TestLoss += loss.item()
                _, pred = t.max(Outputs.data, 1)
                TestAcc += t.mean(pred.eq(Labels.data.view_as(pred)).type(t.FloatTensor)).item() * len(InputData)
            print('Loss: {}\tAcc: {}%'.format(round(TrainLoss, 2),
                                              round(100*TestAcc/(len(testdataloader) * 32), 2)))
            print('-'*60)  
        test_acc_arr.append(100*TestAcc/(len(testdataloader)*32))
        test_loss_arr.append(TestLoss)
    return train_loss_arr, train_acc_arr, test_loss_arr, test_acc_arr

# 进行训练并绘制训练曲线
train_loss_arr, train_acc_arr, test_loss_arr, test_acc_arr = train(resnet, TrainLoader, TestLoader, optimizer, loss)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(train_loss_arr, label='train loss')
ax1.plot(test_loss_arr, label='test loss')
ax1.legend()
ax1.set_title('Loss Curve')
ax1.set_xlabel('epocs')
ax1.set_ylabel('loss')
ax2 = fig.add_subplot(122)
ax2.plot(train_acc_arr, label='train acc')
ax2.plot(test_acc_arr, label='test acc')
ax2.legend()
ax2.set_title('Accuracy Curve')
ax2.set_xlabel('epocs')
ax2.set_ylabel('loss')
plt.show()

# 打印测试集的真实/预测结果
for InputData, Labels in enumerate(TestSet):
    Outputs = resnet(Labels[0].unsqueeze(0))
    _, pred = t.max(Outputs.data, 1)
    pred_name = namedict[str(pred.item())]
    real_name = namedict[str(Labels[1].item())]
    print('real name: {}\t\t\t\tpredict name: {}'.format(real_name, pred_name))
t.save(resnet, r'face+\resnet.pth')