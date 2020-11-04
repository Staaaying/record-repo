# 基于ResNet迁移学习的LFW人脸识别分类

LFW数据集(Labeled Faces in the Wild)是马萨诸塞大学阿姆斯特分校计算机视觉研究所整理制作的一个非限制环境下人脸数据集，包含5749人合计13233张图片，图片大小都是250x250

本代码背景是一份CNN的人脸分类报告，仅需要完成简单的人脸分类即可，不需要完成人脸识别，因此就当作是人脸识别的简单入门，之后的话可能会根据自己的兴趣做一个人脸识别检测的demo程序用在树莓派上面

PS. 基于**pytorch-gpu 1.5.1**实现，但是为了通用性所以改成了cpu版本，需要使用gpu的同学请自行添加相应代码

## 数据集准备

### 下载数据集

可以到[LFW官网](http://vis-www.cs.umass.edu/lfw/)上下载数据集，下载之后会有好几个压缩包，我们只需要其中的**lfw.tgz**文件，解压之后就得到了包含所有图片的文件夹

也可以直接拿我下好的数据集，下面是度娘链接

链接：https://pan.baidu.com/s/152iVUmPoMDQN_B94hJWETA 
提取码：7a6h 
复制这段内容后打开百度网盘手机App，操作更方便哦--来自百度网盘超级会员V5的分享(**炫耀下我的v5的(～￣▽￣)～**)

### 制作DataSet

考虑到LFW原始数据集中有很多人只有一张照片，也有部分名人，像布什这种一个人就有上百张照片，一方面为了保持每个人对应的人脸照片量合适，另一方面尽量减少需要分类的人的个数以减小网络大小方便训练，因此需要从LFW数据集中挑选一部分照片用于本次实验。这里最终挑选的是拥有30-100张照片的这部分人，共有29人，也就是说最终的CNN需要分类的个数为29类，对于小实验而言可以接受了

制作过程分为以下几步：

1. 读取文件夹，获取图片及人名
2. 挑选其中符合要求的人脸图片并将人名转换为整数标签
3. 对人脸图片进行变换后和人名标签一起存入DataSet
4. 定义DataLoader用于后续训练

PS. 在图像处理的时候，因为ResNet的图片输入大小是224x224，因此做了一个中心裁剪

```python
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
```

## 调用ResNet18

pytorch官方提供了很多CNN网络的现成版本可以直接调用，就不用自己费力去写了。而且官方提供的网络都有预训练版本，可以直接拿在ImageNet训练过的CNN网络在我们的简易LFW数据集上稍微训练微调，从而实现迁移学习，效果一般都会比较好。

考虑到我们简易LFW数据集的规模，用**ResNet18**就可以了，把**pretrained**属性设置为**True**使用预训练版本，初始使用的话会自动下载网络参数，需要等一会。ResNet18模型没办法直接运用在我们的数据集上，需要做如下三点变换

1. 将输入图片的大小转为N x C x 224 x 244
2. 将**ResNet18**网络中的**requires_grad**置为False，使其后续不参与训练更新(可设置也可以不设置，看哪个效果好而定，不过不更新ResNet网络参数的话训练更新会更快，但是通常效果会差一些)
3. 将**ResNet18**网络的**fc**分类头改为适合我们数据集的大小

```python
# 调用预训练的resnet18进行迁移学习
# resnet50参数量过多，训练效果不太好
resnet = models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

# 将resnet的输出fc(全连接层)替换为本任务所需的格式
# 1000-->256-->relu-->dropout-->29-->softmax
fc_inputs = resnet.fc.in_features
resnet.fc = nn.Sequential(
    nn.Linear(fc_inputs, 256),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(256, 29)
)
```

## 进行迁移学习

之后的步骤就跟通常的CNN训练没有区别了，设置好参数按照模板进行训练即可，由于迁移学习的效果比较好，因此这里也不需要特别设置网络训练的参数，保持默认即可

```python
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
```

## 模型分类结果

训练完成后模型的分类准确率训练集上差不多99%，测试集上最高可以到90%，还是比较符合预期了，毕竟整个网络其实没有进行太多的调整

![2e0f7c211c9bf84603ad2fcdf3008f3](http://typora-staaaying.oss-cn-chengdu.aliyuncs.com/img/2e0f7c211c9bf84603ad2fcdf3008f3.png)

拿**lfw_test**中的8张人脸照片进行测试，其中6张正确，2张错误，看了下分类错误的两张之一

![Jean_Chretien_0055](http://typora-staaaying.oss-cn-chengdu.aliyuncs.com/img/Jean_Chretien_0055.jpg)![David_Beckham_0024](http://typora-staaaying.oss-cn-chengdu.aliyuncs.com/img/David_Beckham_0024.jpg)

左边是**Jean Chretien(加拿大前总理)**，右边是**大名鼎鼎的贝克汉姆**，网络把总理的人脸照片错误识别成了贝克汉姆。讲道理，有一说一，我觉得没啥毛病，总理也挺帅的😎😎😎

有兴趣的同学也可以了解下总理的故事，还挺励志的。

完整代码GitHub地址：https://github.com/Staaaying/record-repo/face-classfication/resnet