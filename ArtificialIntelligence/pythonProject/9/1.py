from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# 生成分类数据集
X_classification, y_classification = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# 训练决策树分类模型
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train_clf, y_train_clf)

# 输出测试集的性能指标
y_pred_clf = clf.predict(X_test_clf)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print("Classification Accuracy:", accuracy)

# 生成回归数据集
X_regression, y_regression = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# 训练决策树回归模型
reg = DecisionTreeRegressor(random_state=42)
reg.fit(X_train_reg, y_train_reg)

# 输出测试集的性能指标
y_pred_reg = reg.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print("Regression Mean Squared Error:", mse)
