import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 读取数据
data = pd.read_csv('testSet.txt', header=None,sep='\s+')
X = data.iloc[:, :-1]  # 特征
y = data.iloc[:, -1]   # 标签
# print(X)
# print(y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve
import matplotlib.pyplot as plt

# 训练逻辑回归模型
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train, y_train)

# 在训练集上做预测
y_train_pred = log_reg_model.predict(X_train)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 计算准确率
accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy:", accuracy)

# 计算精确率
precision = precision_score(y_train, y_train_pred)
print("Precision:", precision)

# 计算召回率
recall = recall_score(y_train, y_train_pred)
print("Recall:", recall)

# 计算F1-score
f1 = f1_score(y_train, y_train_pred)
print("F1-score:", f1)

# 绘制精确率和召回率相对于阈值的函数图
y_scores = log_reg_model.decision_function(X_train)
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
plt.plot(thresholds, precisions[:-1], label="Precision")
plt.plot(thresholds, recalls[:-1], label="Recall")
plt.xlabel("Threshold")
plt.legend(loc="upper left")
plt.title("Precision and Recall vs. Threshold")
plt.show()

# 绘制ROC曲线图
# 计算 ROC 曲线的参数
fpr, tpr, thresholds = roc_curve(y_train, y_scores)
# 计算 AUC
roc_auc = auc(fpr, tpr)

# 画 ROC 曲线图
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.2f})".format(roc_auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# 输出AUC面积
# auc_score = auc(fpr, tpr)
print("AUC Score:", roc_auc)

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer

# 创建SGDClassifier模型
sgd_model = SGDClassifier()

# 定义评估指标
scoring = {
    'accuracy': [],
    'precision': [],
    'recall': [],
    'f1': []
}
# lst=['accuracy','precision','recall','f1']
# print(lst[0])
# 进行五折交叉验证
for key,value in scoring.items():
    scores = cross_val_score(sgd_model, X_test, y_test, cv=5, scoring=key)
    scoring[key].extend(scores)
# 输出评估指标结果

print("Cross Validation Results:")
for key,value in scoring.items():
    print(f'{key}:{np.array(value).mean()}')
# print("Accuracy:", lst[0].mean())
# print("Precision:", lst[1].mean())
# print("Recall:", lst[2].mean())
# print("F1-score:", lst[3].mean())
