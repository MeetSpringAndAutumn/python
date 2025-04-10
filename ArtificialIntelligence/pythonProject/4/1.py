import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 读取数据
data = np.loadtxt('../3/data_multivar.txt',delimiter=',')
# print(data)
X=data[:,:-1]
y=data[:,-1]
# print(X)
# print(y)


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "SGD Regression": SGDRegressor()
}

# 训练模型并输出结果
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_mse = mean_squared_error(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    results[name] = {
        "Train MSE": train_mse,
        "Train MAE": train_mae,
        "Test MSE": test_mse,
        "Test MAE": test_mae,
        "Test R2": test_r2
    }

# 输出结果
for name, metrics in results.items():
    print(name)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    print()

# 选择一种模型进行十折交叉验证
selected_model = SGDRegressor()
cv_scores_mse = cross_val_score(selected_model, X, y, cv=10, scoring='neg_mean_squared_error')
cv_scores_mae = cross_val_score(selected_model, X, y, cv=10, scoring='neg_mean_absolute_error')
cv_scores_r2 = cross_val_score(selected_model, X, y, cv=10, scoring='r2')

print("Cross-validation results:")
print(f"CV MSE: {-cv_scores_mse.mean()}")
print(f"CV MAE: {-cv_scores_mae.mean()}")
print(f"CV R2: {cv_scores_r2.mean()}")