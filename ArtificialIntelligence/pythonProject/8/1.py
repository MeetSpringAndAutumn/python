import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

# 生成三维二元分类数据集并可视化
X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_clusters_per_class=1,
                           n_informative=2, n_redundant=0, n_repeated=0, random_state=42)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Classification Data')
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建机器学习流水线
pipelines = {
    'LinearSVC': Pipeline([('poly', PolynomialFeatures()), ('scaler', StandardScaler()), ('clf', LinearSVC(dual='auto'))]),
    'SVC_linear': Pipeline([('poly', PolynomialFeatures()), ('scaler', StandardScaler()), ('clf', SVC(kernel='linear'))]),
    'SVC_poly': Pipeline([('poly', PolynomialFeatures()), ('scaler', StandardScaler()), ('clf', SVC(kernel='poly'))]),
    'SVC_rbf': Pipeline([('poly', PolynomialFeatures()), ('scaler', StandardScaler()), ('clf', SVC(kernel='rbf'))])
}

# 训练模型并输出性能报告
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred))

# 结合网格调参寻找最优参数组合
parameters = {
    'poly__degree': [2, 3],
    'clf__C': [0.1, 1, 10],
    'clf__gamma': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipelines['SVC_rbf'], parameters, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 输出最优参数组合和最优模型的性能报告
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("Best Parameters:", best_params)
y_pred = best_model.predict(X_test)
print("Best Model Performance:")
print(classification_report(y_test, y_pred))
