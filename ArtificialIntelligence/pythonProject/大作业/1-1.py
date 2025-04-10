import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# 读取数据
data = pd.read_csv('creditcard.csv')

# 显示数据概况
print(data.info())
print(data.describe())
print(data.head())

# 查看欺诈交易与正常交易的数据量对比，做成饼图，显示百分比
labels = ['Normal', 'Fraud']
sizes = data['Class'].value_counts()
explode = (0, 0.1)  # 仅“Fraud”部分突出显示

plt.figure(figsize=(8, 6))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.2f%%', startangle=140)
plt.axis('equal')
plt.title('Fraud vs Normal Transactions')
plt.show()

# 分别可视化查看正常交易和欺诈交易下交易时间和交易金额的关系
plt.figure(figsize=(12, 6))

# 正常交易
plt.subplot(1, 2, 1)
normal_transactions = data[data['Class'] == 0]
sns.scatterplot(x='Time', y='Amount', data=normal_transactions, alpha=0.6)
plt.title('Normal Transactions: Time vs Amount')
plt.xlabel('Time')
plt.ylabel('Amount')

# 欺诈交易
plt.subplot(1, 2, 2)
fraud_transactions = data[data['Class'] == 1]
sns.scatterplot(x='Time', y='Amount', data=fraud_transactions, alpha=0.6, color='red')
plt.title('Fraud Transactions: Time vs Amount')
plt.xlabel('Time')
plt.ylabel('Amount')

plt.tight_layout()
plt.show()

# 对 V1-V28 个属性进行分析，根据其特征重要性选择其中一半的属性作为特征
# 特征和标签
X = data.drop(['Class', 'Time', 'Amount'], axis=1)
y = data['Class']

# 随机森林模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 计算特征重要性
feature_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)

# 选择一半的重要特征
selected_features = feature_importances.head(len(feature_importances) // 2).index.tolist()

print("Selected Features:", selected_features)

# 按 7：3 的比例将数据集分成训练集和测试集
X = data[selected_features + ['Time', 'Amount']]
y = data['Class']

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 定义评估函数
def print_metrics(model_name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"{model_name} Confusion Matrix:\n{cm}")
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")

# 逻辑回归
lr_model = LogisticRegression(max_iter=5000, random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print_metrics("Logistic Regression", y_test, lr_pred)

# 随机森林
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print_metrics("Random Forest", y_test, rf_pred)

# 支持向量机
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
print_metrics("Support Vector Machine", y_test, svm_pred)

# 选择表现最好的模型进行五折交叉验证（假设随机森林表现最好）
best_model = rf_model
scores = cross_val_score(best_model, X_test, y_test, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy: {scores.mean():.4f}")
