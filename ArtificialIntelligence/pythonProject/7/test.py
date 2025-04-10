import numpy as np
import matplotlib.pyplot as plt
def load_data(input_file):
    X=[]
    y=[]
    with open(input_file,'r') as f:
        for line in f.readlines():
            line=line.strip()
            line=[float(x) for x in line.split(',')]
            X.append(line[:-1])
            y.append(line[-1])
    X=np.array(X)
    y=np.array(y)
    return X,y
X,y=load_data('../7/data_multivar_imbalance.txt')
# 数据可视化
class_0=np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1=np.array([X[i] for i in range(len(X)) if y[i]==1])
plt.figure()
plt.scatter(class_0[:,0],class_0[:,1],facecolors='green',marker='s')
plt.scatter(class_1[:,0],class_1[:,1],facecolors='red',marker='<')
plt.title('class data')
plt.show()
# 划分数据集并用SVC模型训练
from sklearn import model_selection
from sklearn.svm import SVC
import sklearn.metrics as sm
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=5)
# SVC参数https://blog.csdn.net/transformed/article/details/90437821
# gamma参数https://blog.csdn.net/weixin_44943389/article/details/130800550
# params={'kernel':'linear','C':1.0,'probability':False,'tol':0.001}
# params={'kernel':'poly','C':1.0,'degree':3,'gamma':'auto','probability':False,'tol':0.001}
# params={'kernel':'sigmoid','C':1.0,'gamma':'auto','probability':False,'tol':0.001}
params={'kernel':'rbf','C':1.0,'gamma':'auto','probability':False,'tol':0.001}
classifer=SVC(**params)
classifer.fit(X_train,y_train)
y_train_pre=classifer.predict(X_train)
# print("核函数为linear的正确率：",round(sm.accuracy_score(y_train,y_train_pre),2))
# print("核函数为poly的正确率：",round(sm.accuracy_score(y_train,y_train_pre),2))
# print("核函数为sigmoid的正确率：",round(sm.accuracy_score(y_train,y_train_pre),2))
# print("核函数为rbf的正确率：",round(sm.accuracy_score(y_train,y_train_pre),2))
# 画分类模型图
def plot_classifier(classifier, X, y, title='Classifier boundaries', annotate=False):
    # 画分类边界图
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    step_size = 0.01
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)
    plt.figure()
    plt.title(title)
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)
    #画样本点
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.xticks(())
    plt.yticks(())
    if annotate:
        for x, y in zip(X[:, 0], X[:, 1]):
            plt.annotate(
                '(' + str(round(x, 1)) + ',' + str(round(y, 1)) + ')',
                xy = (x, y), xytext = (-15, 15),
                textcoords = 'offset points',
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.6', fc = 'white', alpha = 0.8),
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))
# plot_classifier(classifer, X_train, y_train, 'Training dataset')
# plt.show()
# 输出训练集/测试集的性能报告
y_test_pre=classifer.predict(X_test)
target_name=['class0','calss1']
# print(sm.classification_report(y_test, y_test_pre,target_names=target_name))
# 寻找最优参数https://blog.csdn.net/weixin_39664995/article/details/111215415
# 参数寻优与网格搜索
from sklearn.model_selection import GridSearchCV
X, y = load_data('../7/data_multivar_imbalance.txt')
param_grid={'C':[0.001,0.01,0.1,1,10,100],
            'gamma':[0.001,0.01,0.1,1,10,100]}
gsm=GridSearchCV(SVC(), param_grid,cv=5)
gsm.fit(X,y)
y_test_pre=gsm.predict(X_test)
# 对模型评价的函数、属性
# print(round(gsm.score(X_test,y_test),2)) #模型得分:网格搜索参数寻优方式下的预测精度值
# print(gsm.best_params_) #网格参数寻优方式下的最佳参数取值组合
# print(gsm.best_score_) #网格参数寻优方式下的最佳预测精度值
# print(gsm.best_estimator_) #网格参数寻优方式下的学习模型
# print(gsm.cv_results_) #网格搜索的结果,保存了搜索的所有内容
# 参数寻优与非网格搜索
param_grid2=[{'kernel':['linear'],'C':[1,10,50,600]},
             {'kernel':['poly'],'degree':[2,3],'C':[1,10,50,600]},
             {'kernel':['rbf'],'gamma':[0.1,1,10],'C':[1,10,50]}]
metrics=['precision','recall_weighted']
# for metric in metrics:
#     print("查找最优参数的",metric)
#     classifer=GridSearchCV(SVC(), param_grid2,cv=5,scoring=metric)
#     classifer.fit(X_train,y_train)
#     means=classifer.cv_results_['mean_test_score']
#     params=classifer.cv_results_['params']
#     for mean,param in zip(means,params):
#         print("{}with{}".format(round(mean,2),param))
#     print("最高分数的参数集为：",classifer.best_params_)
#     print("最高分数为：",classifer.best_score_)
#     y_test_pred =classifer.predict(X_test)
#     print("\nFull performance report:\n")
#     print(sm.classification_report(y_test, y_test_pred))

# 基于嵌套交叉验证机制的网格参数寻优
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
#               'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
scores=model_selection.cross_val_score(GridSearchCV(SVC(), param_grid2,cv=5),
                                        X,y,scoring='accuracy',cv=5)
# print("基于嵌套交叉验证机制下的网格搜索预测精度值: ", scores)
# print("基于嵌套交叉验证机制下的网格搜索预测平均精度值: {:.2f}".format(scores.mean()))





# 解决类型数量不平衡问题:params = {'kernel': 'linear','class_weight':'balanced'}
X,y=load_data('data/data_multivar_imbalance.txt')
class_0 = np.array([X[i] for i in range(len(X)) if y[i]==0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i]==1])
plt.figure()
plt.scatter(class_0[:,0], class_0[:,1], facecolors='green', edgecolors='black',
marker='s')
plt.scatter(class_1[:,0], class_1[:,1], facecolors='red', edgecolors='black',
marker='s')
plt.title('Input data')
plt.show()

# 参数class_weight的作用是统计不同类型数据点的数量，调整权重，让类型不平衡问题不影响分类效果。
params = {'kernel': 'linear','class_weight':'balanced'}
params = {'kernel': 'linear','class_weight':'auto'}
classifier=SVC(**params)
classifier.fit(X_train,y_train)
plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()
y_test_pred=classifier.predict(X_test)
plot_classifier(classifier, X_test, y_test, 'Testing dataset')
plt.show()