import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 1. 读取 heart.csv 数据集
heart_data = pd.read_csv("../5/heart.csv")

# 2. 划分训练集和测试集
X = heart_data.drop('target', axis=1)
y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 定义网格调参的参数范围
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 1, 10],
    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}

# 4. 使用网格调参
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出网格调参下的模型得分，最佳参数取值组合，最佳预测精度值，学习模型
print("Grid Search Best Score:", grid_search.best_score_)
print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Best Estimator:", grid_search.best_estimator_)
print("Grid Search Test Score:", grid_search.score(X_test, y_test))

# 5. 输出测试集的性能报告
y_pred = grid_search.predict(X_test)
print("Test Set Performance Report:")
print(classification_report(y_test, y_pred))
