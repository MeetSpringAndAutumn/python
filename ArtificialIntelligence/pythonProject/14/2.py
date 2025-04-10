import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 读取数据
file_path = 'ch15_PinganStock.csv'
data = pd.read_csv(file_path,encoding='gbk')

# 计算EMA15
def calculate_ema(prices, days):
    return prices.ewm(span=days, adjust=False).mean()

data['EMA15'] = calculate_ema(data['收盘'], 15)

# 计算RDP5, RDP10, RDP15, RDP20
def calculate_rdp(prices, days):
    return prices.pct_change(periods=days).shift(-days)

data['RDP5'] = calculate_rdp(data['收盘'], 5)
data['RDP10'] = calculate_rdp(data['收盘'], 10)
data['RDP15'] = calculate_rdp(data['收盘'], 15)
data['RDP20'] = calculate_rdp(data['收盘'], 20)

# 计算目标变量RDP
def calculate_target(prices, days):
    ema_current = prices.ewm(span=3, adjust=False).mean()
    ema_future = prices.shift(-days).ewm(span=3, adjust=False).mean()
    return (ema_future - ema_current) / ema_current

data['RDP'] = calculate_target(data['收盘'], 5)
# print(data.shape)
# 去除缺失值
data = data.dropna()
# print(data.shape)
# 构建输入特征矩阵和目标变量
X = data[['RDP5', 'RDP10', 'RDP15', 'RDP20', 'EMA15']]
y = data['RDP']
# print(X)
# 使用给定时间范围划分训练集和测试集
# train_end_date = '2019-03-07'
# test_start_date = '2019-03-08'
#
l=round(len(data)*0.9)
X_train = X[:l]
y_train = y[:l]
X_test = X[l:]
y_test = y[l:]
# print(X_test.shape)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False,)
# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# 对比预测结果和真实值可视化
plt.figure(figsize=(14, 7))
plt.plot(y_test.values, label='True Values')
plt.plot(y_pred, label='Predicted Values')
plt.legend()
plt.show()
