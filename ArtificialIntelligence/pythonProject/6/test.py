import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors, datasets
from sklearn import linear_model
from sklearn import model_selection

# 多分类训练及评价指标
X = []
y = []
with open('data/data_multivar2.txt', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = line.split(',')
        X.append(line[:-1])
        y.append(line[-1])
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)
# print(X.shape)
# print(np.size(np.unique(y)))
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=5)
classifer = linear_model.LogisticRegression(solver='liblinear', C=1)
classifer.fit(X_train, y_train)
y_train_pre = classifer.predict(X_train)
import sklearn.metrics as sm

print(sm.confusion_matrix(y_train, y_train_pre))
# 画混淆矩阵
# plt.imshow(sm.confusion_matrix(y_train, y_train_pre),interpolation='nearest',cmap=plt.cm.Paired)
# plt.colorbar()
# plt.title('confusion matrix')
# tick_marks=np.arange(4)
# plt.xticks(tick_marks)
# plt.yticks(tick_marks)
# plt.xlabel('true label')
# plt.ylabel('pre label')
# plt.show()
# 多分类的accuracy precision ,recall
# print("准确率：",sm.accuracy_score(y_train,y_train_pre))
# print("marco精确率：",sm.precision_score(y_train,y_train_pre,average='macro'))
# print("macro召回率：",sm.recall_score(y_train,y_train_pre,average='macro'))
# print("weight精确率：",sm.precision_score(y_train,y_train_pre,average='weighted'))
# print("weight召回率：",sm.recall_score(y_train,y_train_pre,average='weighted'))
# 多分类性能报告
y_true = [0, 1, 2, 2, 2]
y_pred = [0, 0, 2, 2, 1]
target_names = ['class0', 'class1', 'class2']
report = sm.classification_report(y_true, y_pred, target_names=target_names)


# print(report)

# 分类模型图
def plot_classifier(classifier, X, y):
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    step_size = 0.01
    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    plt.scatter(x_values, y_values, s=10, color='red')

    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    print(mesh_output[:10])
    mesh_output = mesh_output.reshape(x_values.shape)
    print(mesh_output.shape)
    plt.figure()
    # plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.show()


plot_classifier(classifer, X_train, y_train)