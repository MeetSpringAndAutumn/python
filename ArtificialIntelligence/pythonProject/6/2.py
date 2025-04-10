import numpy as np
import pandas as pd
import sklearn.metrics as ms
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report,make_scorer

# 读取数据
car_df = pd.read_csv('../3/car.txt', header=None)

# 对特征数据进行OneHot编码
X = pd.get_dummies(car_df.iloc[:, :-1])
X1=X
X=np.array(X)
y_original = car_df.iloc[:, -1]

# 统计编码前每个类别的个数
unique_labels, counts = np.unique(y_original, return_counts=True)

print("不同类别样本的个数：")
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")
# 对标签数据进行LabelEncoder编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(car_df.iloc[:, -1])

# (2) 选择一种average评价指标，使用逻辑回归模型训练数据，并输出五折交叉验证的正确率、精确率、召回率和F1分数
# scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, zero_division=0,average='weighted'),
    'recall': make_scorer(recall_score, zero_division=0,average='weighted'),
    'f1': make_scorer(f1_score, zero_division=0,average='weighted'),
}
log_reg = LogisticRegression()

# cv_results = cross_validate(log_reg, X, y, cv=5, scoring=scoring,error_score=0)
for key,value in scoring.items():
    scores = cross_val_score(log_reg, X, y, cv=5, scoring=value)
    scoring[key]=scores
    # print(scores)
accuracy_avg = np.mean(scoring['accuracy'])
precision_avg = np.mean(scoring['precision'])
recall_avg = np.mean(scoring['recall'])
f1_avg = np.mean(scoring['f1'])

print("\n五折交叉验证结果：")
print(f"平均正确率：{accuracy_avg}")
print(f"平均精确率：{precision_avg}")
print(f"平均召回率：{recall_avg}")
print(f"平均F1分数：{f1_avg}")

# (3) 输出五折交叉验证的性能报告
print("\n五折交叉验证性能报告：")
log_reg.fit(X, y)  # 使用全部数据进行训练
# print(X)
y_pred = cross_val_predict(log_reg, X, y, cv=5)  # 预测
print(classification_report(y, y_pred))

# (4) 生成一个测试样本，使用模型进行预测，并输出预测值
# 假设新的测试样本为第一行数据
test_sample = X1.iloc[0, :].values.reshape(1, -1)
predicted_class = log_reg.predict(test_sample)
predicted_class_label = label_encoder.inverse_transform(predicted_class)
print(f"\n测试样本的预测类别为：{predicted_class_label}")
