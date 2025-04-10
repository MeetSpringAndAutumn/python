import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

# 1. 读取data_multivar_imbalance.txt数据
data = pd.read_csv("data_multivar_imbalance.txt", header=None)

# print(data)
# 添加列名
data.columns = ['feature1', 'feature2', 'labels']

# 2. 根据labels对样本前两列特征进行数据可视化
plt.figure(figsize=(10, 6))
for label in data['labels'].unique():
    subset = data[data['labels'] == label]
    plt.scatter(subset['feature1'], subset['feature2'], label=label)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Feature 1 vs Feature 2 by Label')
plt.show()

# 3. 划分训练集和测试集
X = data[['feature1', 'feature2']]
y = data['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 设置SVC模型参数和核函数
params = [
    {'kernel':'linear','C':1.0,'tol':0.001},
    {'kernel':'poly','C':1.0,'degree':3,'gamma':'auto','tol':0.001},
    {'kernel':'sigmoid','C':1.0,'gamma':'auto','tol':0.001},
    {'kernel': 'rbf', 'C': 1.0, 'gamma': 'auto',  'tol': 0.001}
]

# 4. 手动调参，输出不同参数下的测试集的性能报告
for param in params:
    svc = SVC(**param)
    svc.fit(X_train, y_train)
    y_pred = svc.predict(X_test)
    print("Parameters:", param)
    print(classification_report(y_test, y_pred))

# 5. 使用网格调参
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.1, 1, 'scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出网格调参下的模型得分，最佳参数取值组合，最佳预测精度值，学习模型
print("Grid Search Best Score:", grid_search.best_score_)
print("Grid Search Best Parameters:", grid_search.best_params_)
print("Grid Search Best Estimator:", grid_search.best_estimator_)
print("Grid Search Test Score:", grid_search.score(X_test, y_test))
print('\n')
# 6. 使用非网格调参，基于最高分数的参数集模型输出测试集的性能报告
param_grid2=[{'kernel':['linear'],'C':[1,10,50,600]},
             {'kernel':['poly'],'degree':[2,3],'C':[1,10,50,600]},
             {'kernel':['sigmoid'],'C':[1,10,50,600]},
             {'kernel':['rbf'],'gamma':[0.1,1,10],'C':[1,10,50]}]
grid_search2 = GridSearchCV(SVC(), param_grid2, cv=5)
grid_search2.fit(X_train, y_train)
best_params = grid_search2.best_params_
best_svc = SVC(**best_params)
best_svc.fit(X_train, y_train)
y_pred = best_svc.predict(X_test)
print("Best Parameters:", best_params)
print(classification_report(y_test, y_pred))

# 7. 使用嵌套交叉验证机制下的网格参数寻优搜索
nested_predict = cross_val_predict(grid_search, X_test, y_test, cv=5)
# print(nested_predict.mean())
print(classification_report(y_test,nested_predict))
# print("Nested Cross-Validation Score:", nested_score.mean())
