from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 加载digits数据集
digits = load_digits()
X, y = digits.data, digits.target
# print(digits.data.shape)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用PCA进行降维
n_components = 20  # 设置PCA的主成分数量
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 使用分类模型（这里使用随机森林分类器）
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_pca, y_train)

# 在测试集上进行预测
y_pred = rf_classifier.predict(X_test_pca)

# 输出性能指标
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("性能指标：")
print(f"准确率：{accuracy:.2f}")
print("分类报告：")
print(report)

# 使用原始数据进行分类
rf_classifier_raw = RandomForestClassifier(random_state=42)
rf_classifier_raw.fit(X_train, y_train)
y_pred_raw = rf_classifier_raw.predict(X_test)
accuracy_raw = accuracy_score(y_test, y_pred_raw)
report_raw = classification_report(y_test, y_pred_raw)

# 输出原始数据的性能指标
print("原始数据性能指标：")
print(f"准确率：{accuracy_raw:.2f}")
print("分类报告：")
print(report_raw)

# 输出PCA降维后的性能指标
print("\nPCA降维后的性能指标：")
print(f"准确率：{accuracy:.2f}")
print("分类报告：")
print(report)
