import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def RealDemandFunc(price, i, I, a=3000, b=4, c=10, r=100, scale=150):
    '''
    真实需求函数
    '''
    d = a + r + c*(1+i)*price - b*price**2 + np.random.normal(loc=0, scale=scale, size=price.shape)

    return np.minimum(d, I)

def GaussProcess(x, y):
    '''
    根据传入的x，y训练高斯过程回归模型，返回预测函数
    param
    x: K*N price
    y: K*N demand
    return
    包含N个高斯过程评估器的列表 
    '''
    GPList = []
    for i in range(y.shape[1]):
        kernel = Matern()
        # gpr = GaussianProcessRegressor(kernel=kernel)
        gpr = GaussianProcessRegressor()
        gpr.fit(x, y[:, i])
        GPList.append(gpr.predict)
    return GPList

def LinPro(p, d, a, c):
    '''
    根据输入的参数解算线性规划问题
    Params：
    p: K*N 矩阵
    d: K*N 矩阵
    a: N*G 向量，产品与资源之间的消耗系数
    c: G*1 向量，当前各个资源的约束
    return:
    概率x,x>0的条件已经默认设置在linprog里面
    '''
    c_coffe = np.sum(p*d, axis=1)
    A_ub = np.concatenate(((d.dot(a)).T, np.ones((1, p.shape[0]))), axis=0)
    b_ub = np.concatenate((c, np.ones(1)), axis=0)
    return linprog(-c_coffe, A_ub=A_ub, b_ub=b_ub).x


# --------------------------------------------------
# # 初始参数定义
# 初始价格,K*N矩阵
price = np.array([
    [1.0],
    [6.0],
    [11.0],
    [16.0],
    [21.0]])
price = np.array([
    [1.0],
    [12.5],
    [25.0],
    [37.5],
    [50.0]])
price = np.tile(price, (1, 4))
# 其他常数
T = 250
K = price.shape[0]
N = price.shape[1]
alpha = 1200

# 库存系数
a_coeff = np.eye(N)

# 库存及库存约束
I = alpha * T * np.ones((N))
c = I / T

# 需求先验
demand = price*5


price_GP = price
demand_GP = demand
ratio = []
ratio_temp = []
# -------------------BZ-------------------
M = 1
interval = int(T ** (2 / 3))    # BZ算法中的观察期
for t in range(T): 
    if t <= interval:
    # 建立高斯过程模型
        GPList = GaussProcess(price_GP, demand_GP)

    # 根据GP，对每个价格进行采样
    demand_temp = np.zeros((K, N))  # 存放GPs采样后的需求矩阵
    prob_price = []     # 存放M次价格抽样的结果
    prob_demand = []    # 存放M次价格抽样后的需求结果
    for m in range(M):
        for k in range(K):
            for idx, GPs in enumerate(GPList):
                GPs_demand = GPs(price[k].reshape(1, -1))
                demand_temp[k, idx] = max(GPs_demand, 0)
        idx = np.argmax(LinPro(price, demand_temp, a_coeff, c))
        prob_price.append(price[idx])
        prob_demand.append(np.maximum(np.minimum(RealDemandFunc(price[idx], t, I), c), np.zeros((N,))))
        if 0 not in prob_demand[-1]:
            ratio_temp.append(demand_temp[idx] / prob_demand[-1])
    if ratio_temp != []:
        ratio.append(np.array(ratio_temp).mean(axis=0))
    # 更新用于构建GPs的价格-需求数组
    price_GP = np.concatenate((price_GP, np.array(prob_price)), axis=0)
    demand_GP = np.concatenate((demand_GP, np.array(prob_demand)), axis=0)

    # 更新库存和库存约束
    c = I/(T-t)
    I = I - np.array(prob_demand).mean(axis=0)
try:
    ratio = np.array(ratio).mean(axis=1)
except np.AxisError:
    ratio = np.array(ratio)

k1, = plt.plot(ratio[ratio<=1], 'r-')
res = ratio[ratio<=1]
plt.scatter(np.arange(res.shape[0])[::15], res[::15], c='r')

# -------------------GPTS-------------------
M=1
I = alpha * T * np.ones((N))
c = I / T
price_GP = price
demand_GP = demand
ratio = []
ratio_temp = []
for t in range(T): 
    # 建立高斯过程模型
    GPList = GaussProcess(price_GP, demand_GP)

    # 根据GP，对每个价格进行采样
    demand_temp = np.zeros((K, N))  # 存放GPs采样后的需求矩阵
    prob_price = []     # 存放M次价格抽样的结果
    prob_demand = []    # 存放M次价格抽样后的需求结果
    for m in range(M):
        for k in range(K):
            for idx, GPs in enumerate(GPList):
                demand_temp[k, idx] = max(GPs(price[k].reshape(1, -1)), 0)
        idx = np.argmax(LinPro(price, demand_temp, a_coeff, c))
        prob_price.append(price[idx])
        prob_demand.append(np.maximum(np.minimum(RealDemandFunc(price[idx], t, I), c), np.zeros((N,))))
        if 0 not in prob_demand[-1]:
            ratio_temp.append(demand_temp[idx] / prob_demand[-1])
    if ratio_temp != []:
        ratio.append(np.array(ratio_temp).mean(axis=0))

    # 更新用于构建GPs的价格-需求数组
    price_GP = np.concatenate((price_GP, np.array(prob_price)), axis=0)
    demand_GP = np.concatenate((demand_GP, np.array(prob_demand)), axis=0)

    # 更新库存和库存约束
    c = I/(T-t)
    I = I - np.array(prob_demand).mean(axis=0)
try:
    ratio = np.array(ratio).mean(axis=1)
except np.AxisError:
    ratio = np.array(ratio)
k2, = plt.plot(ratio[ratio<=1], 'g--')
res = ratio[ratio<=1]
plt.scatter(np.arange(res.shape[0])[::15], res[::15], c='g')


# -------------------GPTS-------------------
M=5
I = alpha * T * np.ones((N))
c = I / T
price_GP = price
demand_GP = demand
ratio = []
ratio_temp = []
for t in range(T): 
    # 建立高斯过程模型
    GPList = GaussProcess(price_GP, demand_GP)

    # 根据GP，对每个价格进行采样
    demand_temp = np.zeros((K, N))  # 存放GPs采样后的需求矩阵
    prob_price = []     # 存放M次价格抽样的结果
    prob_demand = []    # 存放M次价格抽样后的需求结果
    for m in range(M):
        for k in range(K):
            for idx, GPs in enumerate(GPList):
                demand_temp[k, idx] = max(GPs(price[k].reshape(1, -1)), 0)
        idx = np.argmax(LinPro(price, demand_temp, a_coeff, c))
        prob_price.append(price[idx])
        prob_demand.append(np.maximum(np.minimum(RealDemandFunc(price[idx], t, I), c), np.zeros((N,))))
        if 0 not in prob_demand[-1]:
            ratio_temp.append(demand_temp[idx] / prob_demand[-1])
    if ratio_temp != []:
        ratio.append(np.array(ratio_temp).mean(axis=0))

    # 更新用于构建GPs的价格-需求数组
    price_GP = np.concatenate((price_GP, np.array(prob_price)), axis=0)
    demand_GP = np.concatenate((demand_GP, np.array(prob_demand)), axis=0)

    # 更新库存和库存约束
    c = I/(T-t)
    I = I - np.array(prob_demand).mean(axis=0)
try:
    ratio = np.array(ratio).mean(axis=1)
except np.AxisError:
    ratio = np.array(ratio)
k3, = plt.plot(ratio[ratio<=1], 'b-.')
res = ratio[ratio<=1]
plt.scatter(np.arange(res.shape[0])[::15], res[::15], c='b')


plt.ylim([0, 1])
plt.legend([k1, k2, k3], ['BZ', 'GPTS', 'GP-PTS'])
plt.show()