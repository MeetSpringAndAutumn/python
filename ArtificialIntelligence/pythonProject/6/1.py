import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. 读取iris数据
iris_df = pd.read_csv('../iris.csv')

# 划分特征和标签
X = iris_df.iloc[:, :-1].values
y = iris_df.iloc[:, -1].values
scaler=StandardScaler()
X=scaler.fit_transform(X)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# (1) 使用逻辑回归模型训练数据，画出训练集的混淆矩阵图
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_train = log_reg.predict(X_train)
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

plt.imshow(conf_matrix_train, cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.xticks(np.arange(len(iris_df['Species'].unique())), iris_df['Species'].unique())
plt.yticks(np.arange(len(iris_df['Species'].unique())), iris_df['Species'].unique())
plt.title('Confusion Matrix - Training Set')
plt.show()

# (2) 分别输出测试集在average='macro'和average='weighted'时的精确率、召回率和F1分数
from sklearn.metrics import precision_recall_fscore_support

def print_metrics(y_true, y_pred, average):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
    print(f'Precision ({average}): {precision}')
    print(f'Recall ({average}): {recall}')
    print(f'F1-score ({average}): {f1}')

print_metrics(y_test, log_reg.predict(X_test), average='macro')
print_metrics(y_test, log_reg.predict(X_test), average='weighted')

# (3) 输出测试集预测的性能报告
print(classification_report(y_test, log_reg.predict(X_test)))

# # (4) 分别画正则参数C=100 和C=1时的分类器模型图
def plot_decision_boundary(classifier, X, y, title):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)
    # print(x_values)
    plt.figure()
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.title(title)
    plt.show()


C_values = [100, 1]
for C in C_values:
    log_reg = LogisticRegression(C=C)
    log_reg.fit(X_train[:, :2], y_train)  # 使用前两维特征
    plot_decision_boundary(log_reg,X_train[:, :2],y_train   , f'Logistic Regression (C={C})')
