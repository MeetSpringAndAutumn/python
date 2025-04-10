import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.font_manager as fm

# 设置字体为 SimHei，以支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 导入数据
data = pd.read_csv('Churn-Modelling-new.csv',sep=',')
# print(data.info)

data.dropna(inplace=True)

# print("缺失值检查:\n", data.isna().sum())
# 删除无意义列
data.drop(['RowNumber', 'CustomerId', 'Surname',], axis=1, inplace=True)

# 使用LabelEncoder将非数值特征转化为数值特征
le_geography = LabelEncoder()
le_gender = LabelEncoder()

data['Geography'] = le_geography.fit_transform(data['Geography'])
data['Gender'] = le_gender.fit_transform(data['Gender'])

# 信用分数离散化
data['CreditScore'] = pd.cut(data['CreditScore'], bins=[0, 584, 718, float('inf')], labels=[0, 1, 2])

# 年龄离散化
data['Age'] = pd.cut(data['Age'], bins=[0, 20, 40, float('inf')], labels=[0, 1, 2])

# 存贷款情况离散化
# print((data['Balance'] == 0).sum())
# print(data['Balance'].describe())
data['Balance'] = pd.cut(data['Balance'], bins=[-1, 48000, 97198, float('inf')], labels=[0, 1, 2])
# print(data['Balance'].unique())
# 估计收入离散化
data['EstimatedSalary'] = pd.cut(data['EstimatedSalary'], bins=[0, 51002, 149388, float('inf')], labels=[0, 1, 2])

# 分离特征与标签
X = data.drop('Exited', axis=1)
y = data['Exited']

# 使用欠采样方法处理类别不平衡问题
class_counts = data['Exited'].value_counts()
# print("每个类别的样本数量:\n", class_counts)

# 假设标签0的数量更多，即 class_0 是多数类，class_1 是少数类
if class_counts[0] > class_counts[1]:
    majority_class = 0
    minority_class = 1
else:
    majority_class = 1
    minority_class = 0

# 提取不同类别的样本
class_majority = data[data['Exited'] == majority_class]
class_minority = data[data['Exited'] == minority_class]

# 下采样
class_majority_under = class_majority.sample(len(class_minority))
data_under = pd.concat([class_majority_under, class_minority], axis=0)

# 分离特征和标签
X = data_under.drop('Exited', axis=1)
y = data_under['Exited']


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 决策树模型
dt_model = DecisionTreeClassifier(criterion="gini",max_depth=6,min_samples_split=200,random_state=42)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_model.predict_proba(X_test)[:, 1])
roc_auc_dt = auc(fpr_dt, tpr_dt)

print("决策树模型分类报告:\n", classification_report(y_test, y_pred_dt, zero_division=1))

# 绘制决策树混淆矩阵
plt.figure()
plt.matshow(conf_matrix_dt, cmap=plt.cm.Blues)
plt.title('决策树混淆矩阵')
plt.colorbar()
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 绘制决策树ROC曲线
plt.figure()
plt.plot(fpr_dt, tpr_dt, color='darkorange', lw=2, label='ROC曲线 (area = %0.2f)' % roc_auc_dt)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('决策树ROC曲线')
plt.legend(loc="lower right")
plt.show()

# print(X.isna().sum())
# SVM模型

# print(X.shape)
svm_model = SVC(C=10,gamma=0.1,probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_model.predict_proba(X_test)[:, 1])
roc_auc_svm = auc(fpr_svm, tpr_svm)

print("SVM模型分类报告:\n", classification_report(y_test, y_pred_svm, zero_division=1))

# 绘制SVM混淆矩阵
plt.figure()
plt.matshow(conf_matrix_svm, cmap=plt.cm.Blues)
plt.title('SVM混淆矩阵')
plt.colorbar()
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

# 绘制SVM ROC曲线
plt.figure()
plt.plot(fpr_svm, tpr_svm, color='darkorange', lw=2, label='ROC曲线 (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率')
plt.ylabel('真阳性率')
plt.title('SVM ROC曲线')
plt.legend(loc="lower right")
plt.show()

# 决策树交叉验证
dt_scores = cross_val_score(dt_model, X, y, cv=5)
dt_scores_10 = cross_val_score(dt_model, X, y, cv=10)
dt_scores_15 = cross_val_score(dt_model, X, y, cv=15)

# SVM交叉验证
svm_scores = cross_val_score(svm_model, X, y, cv=5)
svm_scores_10 = cross_val_score(svm_model, X, y, cv=10)
svm_scores_15 = cross_val_score(svm_model, X, y, cv=15)

# 绘制交叉验证结果
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.plot(range(1, 6), dt_scores, marker='o', label='决策树')
plt.plot(range(1, 6), svm_scores, marker='x', label='SVM')
plt.xlabel('折数')
plt.ylabel('准确率')
plt.title('5折交叉验证准确率')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(range(1, 11), dt_scores_10, marker='o', label='决策树')
plt.plot(range(1, 11), svm_scores_10, marker='x', label='SVM')
plt.xlabel('折数')
plt.ylabel('准确率')
plt.title('10折交叉验证准确率')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(range(1, 16), dt_scores_15, marker='o', label='决策树')
plt.plot(range(1, 16), svm_scores_15, marker='x', label='SVM')
plt.xlabel('折数')
plt.ylabel('准确率')
plt.title('15折交叉验证准确率')
plt.legend()

plt.tight_layout()
plt.show()
