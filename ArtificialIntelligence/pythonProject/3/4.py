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

# 预测测试集结果
y_test_pred = model.predict(X_test)
# 计算测试集的均方差MSE
mse_test = mean_squared_error(y_test, y_test_pred)

# 计算测试集的平均绝对误差MAE
mae_test = mean_absolute_error(y_test, y_test_pred)
# 输出测试集的均方差MSE
print("测试集的均方差MSE：", mse_test)

# 输出测试集的平均绝对误差MAE
print("测试集的平均绝对误差MAE：", mae_test)