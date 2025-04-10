from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# 获取加州住房数据集
housing = fetch_california_housing()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# 线性模型
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
train_pred_linear = linear_model.predict(X_train)
test_pred_linear = linear_model.predict(X_test)
train_mse_linear = mean_squared_error(y_train, train_pred_linear)
train_mae_linear = mean_absolute_error(y_train, train_pred_linear)
test_mse_linear = mean_squared_error(y_test, test_pred_linear)
test_mae_linear = mean_absolute_error(y_test, test_pred_linear)
test_r2_linear = r2_score(y_test, test_pred_linear)

print("Linear Model Results:")
print(f"Train MSE: {train_mse_linear}")
print(f"Train MAE: {train_mae_linear}")
print(f"Test MSE: {test_mse_linear}")
print(f"Test MAE: {test_mae_linear}")
print(f"Test R2: {test_r2_linear}")

# 岭回归
alpha_values = [0.1, 1, 10]  # 测试不同的alpha参数
ridge_results = {}
for alpha in alpha_values:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    train_pred = ridge.predict(X_train)
    test_pred = ridge.predict(X_test)
    train_mse = mean_squared_error(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    ridge_results[alpha] = {
        "Train MSE": train_mse,
        "Train MAE": train_mae,
        "Test MSE": test_mse,
        "Test MAE": test_mae,
        "Test R2": test_r2
    }

print("\nRidge Regression Results:")
for alpha, metrics in ridge_results.items():
    print(f"Alpha: {alpha}")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()

# 选择模型
selected_model = linear_model

# 十折交叉验证
cv_scores_mse = cross_val_score(selected_model, housing.data, housing.target, cv=10, scoring='neg_mean_squared_error')
cv_scores_mae = cross_val_score(selected_model, housing.data, housing.target, cv=10, scoring='neg_mean_absolute_error')
cv_scores_r2 = cross_val_score(selected_model, housing.data, housing.target, cv=10, scoring='r2')

print("\nCross-validation results:")
print(f"CV MSE: {-cv_scores_mse.mean()}")
print(f"CV MAE: {-cv_scores_mae.mean()}")
print(f"CV R2: {cv_scores_r2.mean()}")
print()
# 保存模型
joblib.dump(selected_model, 'california_housing_model.pkl')

# 从文件中加载模型
model = joblib.load('california_housing_model.pkl')

sample = X_train[1]

# 使用加载的模型进行预测
prediction = model.predict(sample.reshape(1, -1))
# print(y_train[1])
# 打印预测结果
print("Predicted housing price:", prediction)

