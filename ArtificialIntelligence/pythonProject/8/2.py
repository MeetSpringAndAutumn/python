from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 导入数据集
digits = load_digits()
X, y = digits.data, digits.target

# 特征选择
selector = SelectKBest(chi2, k=20)  # 选择得分最高的20个特征
# X_selected = selector.fit_transform(X, y)
selector.fit(X, y)
selected_feature_indices = selector.get_support(indices=True)
feature_scores = selector.scores_

# 输出特征得分和选择后的特征列索引
print("Feature Scores:")
for i, score in enumerate(feature_scores):
    print(f"Feature {i}: {score}")
print("\nSelected Feature Indices:", selected_feature_indices)

# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 构建机器学习流水线
pipeline = Pipeline([
    ("selector",SelectKBest(chi2, k=20)),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale'))  # 参数可以自行调整
])

# 训练模型并输出性能报告
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
