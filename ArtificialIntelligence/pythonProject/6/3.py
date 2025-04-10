import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from mlxtend.plotting import plot_decision_regions
import seaborn as sns

# 读取数据
data = pd.read_csv("data_nn_classifier.txt")

# 提取特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练KNN分类器
knn = KNeighborsClassifier()
knn.fit(X, y)

# 任务(1): 输出精确率、召回率和F1分数
for average in ['macro', 'weighted']:
    y_pred = knn.predict(X)
    precision = precision_score(y, y_pred, average=average)
    recall = recall_score(y, y_pred, average=average)
    f1 = f1_score(y, y_pred, average=average)
    print(f"Average: {average}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# 任务(2): 绘制混淆矩阵图和性能报告
y_pred = knn.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# print("Confusion Matrix:")
# print(conf_matrix)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 任务(3): 绘制KNN分类模型图
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=knn, legend=2)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Classifier Decision Regions')
plt.show()

# 任务(4): 输入新测试样本，输出预测值，并绘制KNN分类结果图
# 假设新测试样本为X_new
X_new = np.array([[1.5, 2.5]])  # 请根据实际情况修改新测试样本的值
y_new_pred = knn.predict(X_new)
print("Predicted Class for New Sample:", y_new_pred)

from sklearn.neighbors import NearestNeighbors

# 找到新样本的K个最近邻点
k = 5  # 假设K值为5
neigh = NearestNeighbors(n_neighbors=k)
neigh.fit(X)
distances, indices = neigh.kneighbors(X_new)
label = 'New Sample (Predicted to be' + str(y_new_pred) + ')'
# print(indices)
# 绘制KNN分类结果图
plt.figure(figsize=(10, 6))
# plot_decision_regions(X, y, clf=knn, legend=0, scatter_kwargs={'alpha': 0, 'label': None})
plot_decision_regions(X, y, clf=knn, legend=2)
l1 = plt.scatter(X_new[:, 0], X_new[:, 1], c='r', marker='x', label=label)
l2 = plt.scatter(X[indices[0], 0], X[indices[0], 1], c='g', marker='o', label='Nearest Neighbors')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KNN Classifier Decision Regions with New Sample and Nearest Neighbors')
# plt.get_legend().remove()
# plt.legend([l1, l2], [label, 'Nearest Neighbors'])
plt.legend()
plt.show()

