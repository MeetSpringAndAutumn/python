from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np

# 读取数据
data = np.loadtxt('data_multivar.txt', delimiter=',')

# 划分特征和标签
X = data[:, :-1]
y = data[:, -1]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性回归模型
model = LinearRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测训练集结果
y_train_pred = model.predict(X_train)

# 计算训练集的均方差MSE
mse = mean_squared_error(y_train, y_train_pred)

# 计算训练集的平均绝对误差MAE
mae = mean_absolute_error(y_train, y_train_pred)

# 计算R方得分
r2 = r2_score(y_train, y_train_pred)

# 计算解释方差
explained_variance = explained_variance_score(y_train, y_train_pred)

# 输出训练集的预测结果
print("训练集的预测结果：")
print(y_train_pred)

# 输出训练集的均方差MSE
print("训练集的均方差MSE：", mse)

# 输出训练集的平均绝对误差MAE
print("训练集的平均绝对误差MAE：", mae)

# 输出R方得分
print("R方得分：", r2)

# 输出解释方差
print("解释方差：", explained_variance)
