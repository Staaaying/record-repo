import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier

file = pd.read_csv(r'ML\failure\predictive_maintenance_case.csv')

# 特征分布可视化
# 0: 124388     1: 106

for metric, data in file.iteritems():
    if 'metric' not in metric:
        continue
    idx = int('33' + metric[-1])
    print('{}\t\tmean: {:.2f}\tstd: {:.2f}'.format(metric, data.mean(), data.std()))
    plt.scatter(np.arange(data.shape[0]), data)
    plt.title(metric)
    plt.show()
# plt.scatter(file['metric4'].to_numpy(), file['metric7'].to_numpy())
# plt.show()
# plt.scatter(file['metric5'].to_numpy(), file['metric6'].to_numpy())
# plt.show()
# plt.scatter(file['metric3'].to_numpy(), file['metric4'].to_numpy())
# plt.show()


# 从dataframe中取出数据
label = file['failure'].to_numpy()
row_list = []
for item in range(9):
    # 7 8列重复因此跳过metric7
    if item == 7:
        continue
    row_list.append('metric' + str(item+1))
data = file[row_list].to_numpy()

# 特征归一化-->Min-Max以及Z-score
minmax = MinMaxScaler()
data = zscore.fit_transform(data)
zscore = StandardScaler()
data = minmax.fit_transform(data)

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
    

pred_bagging = []
pred_vote = []
for item in pred_list:
    vote_tmp_0 = []
    vote_tmp_1 = []
    for classifer in item:
        #  vote_tmp_0，vote_tmp_1用于分别存放unfailure,failure测试集分类结果
        vote_tmp_0.append(classifer.predict(test_data_0))
        vote_tmp_1.append(classifer.predict(test_data_1))
        # 统计分类准确率并存入pred_bagging列表中
        unfailure = (vote_tmp_0[-1] == test_label_0).sum()
        failure = (vote_tmp_1[-1] == test_label_1).sum()
        pred_bagging.append([unfailure/test_data_0.shape[0], failure/test_data_1.shape[0]])
    # 对比一组7个分类模型的分类结果，采用投票方式选择其中占多数的分类结果作为最终样本
    vote_tmp_0 = np.round(np.mean(vote_tmp_0, axis=0))
    vote_tmp_1 = np.round(np.mean(vote_tmp_1, axis=0))
    unfailure = (vote_tmp_0 == test_label_0).sum()
    failure = (vote_tmp_1 == test_label_1).sum()
    pred_vote.append([unfailure/test_data_0.shape[0], failure/test_data_1.shape[0]])

print('Bagging\t\tunfailure acc:{:.2f}({}/{}) failure acc:{:.2f}({}/{})'.format(*np.mean(np.array(pred_bagging), axis=0).round(2)))
print('Vote\t\t\tunfailure acc:{:.2f}({}/{}) failure acc:{:.2f}({}/{})'.format(*np.mean(np.array(pred_vote), axis=0).round(2)))