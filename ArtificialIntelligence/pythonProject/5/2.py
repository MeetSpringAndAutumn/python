import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, \
    precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

# 1. 加载数据集并进行预处理
data = pd.read_csv('heart.csv')
# print(data)
# 处理缺失值
data.dropna(inplace=True)

# 分割特征和标签
X = data.drop('target', axis=1)
y = data['target']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 定义函数来训练模型并评估性能
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # 在训练集上训练模型
    model.fit(X_train, y_train)

    # 在训练集上进行预测
    y_pred_train = model.predict(X_train)

    # 计算评估指标
    cm = confusion_matrix(y_train, y_pred_train)
    acc = accuracy_score(y_train, y_pred_train)
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)

    # 绘制ROC曲线
    y_score = model.predict_proba(X_train)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_train, y_score)
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.title("Precision and Recall vs. Threshold")
    plt.show()


    fpr, tpr, thresholds = roc_curve(y_train, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 输出结果
    print("Confusion Matrix:\n", cm)
    print("Accuracy:", acc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", roc_auc)


# 2. 训练和评估模型
# 逻辑回归模型
logistic_model = LogisticRegression()
print("Logistic Regression Model:")
evaluate_model(logistic_model, X_train, X_test, y_train, y_test)


# 3. 预测一个样本并与真实值比较
sample = X_test[0].reshape(1, -1)
print("\nSample Prediction:")
print("Predicted:", logistic_model.predict(sample))
print("True Value:", y_test.iloc[0])

# 4. 使用五折交叉验证评估模型性能
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scoring = ['accuracy', 'precision', 'recall', 'f1']
logistic_scores = cross_validate(logistic_model, X_scaled, y, cv=kf, scoring=scoring)

# print(logistic_scores)
print("\nCross-validated Logistic Regression Scores:")
for score in scoring:
    print(score.capitalize(), ":", np.mean(logistic_scores['test_' + score]))


