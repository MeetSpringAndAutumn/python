from sklearn.datasets import fetch_california_housing
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 导入数据集
housing = fetch_california_housing()
X, y = housing.data, housing.target
# print(X.shape)
# 特征选择
selector = SelectKBest(f_regression, k=4)  # 选择与目标变量最相关的 10 个特征
selector.fit(X, y)
selected_feature_indices = selector.get_support(indices=True)
feature_scores = selector.scores_

# 输出特征得分和选择后的特征列索引
print("Feature Scores:")
for i, score in enumerate(feature_scores):
    print(f"Feature {i}: {score}")
print("\nSelected Feature Indices:", selected_feature_indices)

# 获取特征名称
selected_feature_names = [housing.feature_names[i] for i in selected_feature_indices]
print("\nSelected Feature Names:", selected_feature_names)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建机器学习流水线
pipeline = Pipeline([
    ("selector",SelectKBest(f_regression, k=4)),
    ('svm', SVR(kernel='rbf', C=1.0, gamma='scale'))  # 参数可以自行调整
])

# 训练模型并预测
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# 计算 MSE 和 MAE
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("\nMean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
