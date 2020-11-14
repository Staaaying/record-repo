# 类别不平衡的设备故障预测

## 背景&需求

数据集是一个设备故障的数据集，记录了近1000台设备在2001-2011年间多个时间节点的设备状态和是否故障，大概数据如下所示，一共有约12W条数据

![image-20201114204344547](http://typora-staaaying.oss-cn-chengdu.aliyuncs.com/img/image-20201114204344547.png)

需求的话就是根据这些数据建立一个设备故障预测模型，尽可能减小failure样本的误判率

难点主要是在**类别极度不平衡**，虽然有12W条数据，但是failure样本数据只有100多条，所以并不是简单调个sklearn就能解决问题。

## 基本思路

对于类别不平衡的机器学习问题，常规方法是采用**过采样**、**降采样**结合的方法对数据进行认为的平衡处理，比如从多数样本中降采样，从中使用随机选取或者聚类选取等方法选出一个子集进行训练，还可以对少数样本使用**SMOTE**方法进行过采样以增加其数据量。

### SMOTE方法

SMOTE(Synthetic Minority OverSampling Technique，合成少数类过采样方法)是一种随机过采样算法的改进，其基本思想不是简单复制少数类中的样本，而是基于对少数类样本进行分析，然后人工合成出一些基于少数类样本生成的新少数类样本，基本过程如下：

1. 对于少数类中的样本x，以欧式距离为量度计算其k近邻
2. 根据提前设定的过采样比率，对于样本x，从k近邻中选择若干个样本xi
3. 选取特征空间中样本x与样本xi连线上某点进行样本合成

![img](http://typora-staaaying.oss-cn-chengdu.aliyuncs.com/img/20190716095433886.png)

SMOTE算法基本思想很简单，也有一些其他改进方法，这里不再赘述，有兴趣的同学可以看一下这篇[知乎专栏](https://zhuanlan.zhihu.com/p/44055312)

### Bagging模型

SMOTE虽好但是并不能解决问题，因为过采样出的样本，不管用什么精巧的方法都不能增加其信息总量，因此不管生成多少样本，少类样本中的信息其实是不变的，因此人工合成适量样本可以帮助模型学习，生成太多的话就会造成过拟合了。

回到本次的问题中，120000:100的样本比例，单纯只靠SMOTE和降采样显然是不够的，因此需要考虑引入其他机制。

本文中使用的方法是bagging模型融合策略，即训练多个弱分类器，然后综合这多个弱分类器的分类结果来得到最终结果，对应到本文中的问题就是从多类样本中降采样一个子集后和少类样本组成训练集进行模型训练，然后重复这个过程多次，最后做一个bagging。

### 数据处理

把数据集中的几个特征分别可视化观察一下可以发现，metric1是均匀分布，metric7和metric8重复，其余的metric除了5，6之外都有比较离散的分布，因此决定采用先Z-score归一化再做一个Min-Max归一化的数据处理

PS. 进一步处理的话，可以从date中做一个累计运行时间特征出来，也可以把设备作为一个特征输入，但是本文中为了简便就没有对这两点进行实现，感兴趣的同学可以自己实践一下。

![Figure_1](http://typora-staaaying.oss-cn-chengdu.aliyuncs.com/img/Figure_1.png)

### 最终方案

最终确定的整体方法如下：

1. Z-score+Min-Max组合数据归一化
2. 从多类样本unfailure中随机选取200组数据，和所有的106组failure数据构成数据集
3. 对上述数据集进行训练集、测试集分割，使用训练集分别训练SVM，决策树，随机森林，线性模型等7个弱分类器
4. 重复2，3步骤M次
5. 对得到的M*7个弱分类器进行bagging融合，采用投票方式在测试集上得到最终的预测结果

## 代码实现

限于篇幅，贴部分代码如下，完整代码[Github地址](https://github.com/Staaaying/record-repo/tree/main/imbalance-failure/failure.py)

```python
# 选取unfailure样本，并将其分割为训练集与测试集
idx = label==0
data0, label0 = data[idx], label[idx]
data0, test_data_0, label0, test_label_0 = train_test_split(data0, label0, test_size=0.0002)

# 选取failure样本，并将其分割为训练集与测试集
idx = label==1
data1, label1 = data[idx], label[idx]
train_data_1, test_data_1, train_label_1, test_label_1 = train_test_split(data1, label1, test_size=0.2)

# pred_list用于存放预测模型，num为循环次数
pred_list = []
num = 100
for _ in range(num):
    # 从unfailure训练集中随机选出150个样本
    # 通过SMOTE上采样，将failure训练集样本增加到150
    idx = np.random.choice(data0.shape[0], 150)
    tmp_data, tmp_label = np.vstack((data0[idx], train_data_1)), np.concatenate((label0[idx], train_label_1))
    smo = SMOTE(sampling_strategy={0:150, 1:150})
    tmp_data, tmp_label = smo.fit_sample(tmp_data, tmp_label)
    
    # 分别训练分类模型
    svc = SVC()
    svc.fit(tmp_data, tmp_label)
    
    nusvc = NuSVC()
    nusvc.fit(tmp_data, tmp_label)
    
    lrsvc = LinearSVC()
    lrsvc.fit(tmp_data, tmp_label)
    
    dtree = DecisionTreeClassifier()
    dtree.fit(tmp_data, tmp_label)
    
    Rtree = RandomForestClassifier()
    Rtree.fit(tmp_data, tmp_label)
    
    clf = SGDClassifier()
    clf.fit(tmp_data, tmp_label)
    
    ada = AdaBoostClassifier()
    ada.fit(tmp_data, tmp_label)
    
    pred_list.append([svc, nusvc, lrsvc, dtree, Rtree, clf, ada])
```

## 最终结果

![image-20201114212818719](http://typora-staaaying.oss-cn-chengdu.aliyuncs.com/img/image-20201114212818719.png)

最后的结果只能说还可以接受吧，在unfailure上自不必说，效果肯定不差，但是在failure上，由于样本实在是太少，所以也只能达到60%左右的分类正确率了。