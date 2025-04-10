# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:00:01 2023

@author: lch15
"""
# 案例：银行客户流失预测
# 数据来源：匿名处理后的国外银行真实数据，共有14个特征
# 数据预处理
# 1\非特征值处理-编码
# Geography与Gender两列为非数值类型特征，需将其转换为数值型特征
import pandas as pd


def quantification(dataPath, outputPath):
    df = pd.read_csv(dataPath)
    x = pd.factorize(df['Geography'])
    y = pd.factorize(df['Gender'])
    df['Geography'] = x[0]
    df['Gender'] = y[0]
    df.to_csv(outputPath)


quantification("Churn-Modelling-new.csv", "Churn-Modelling-newT.csv")

# 2\连续性变量离散化处理
# 决策树算法需要处理离散化的数据，由于原数据集中存在信用分数、年龄、存贷款情况、估计收入等连续型变量，
# 因此需要将这些连续型变量首先转化为离散型变量。
import numpy as np


def discretization(dataPath, outputPath):
    df = pd.read_csv(dataPath)
    CreditScore = [];
    Age = [];
    Balance = [];
    EstimatedSalary = [];
    Exited = []
    # dispersed credit: 0-'low' 1-'middle' 2-'high'
    for i in range(len(df)):
        if df["CreditScore"][i] < 584:
            CreditScore.append(0)
        elif df["CreditScore"][i] < 718:
            CreditScore.append(1)
        else:
            CreditScore.append(2)
    df["CreditScore"] = CreditScore
    # dispersed age: 0-'teenager' 1-'middle age' 2-'old'
    for i in range(len(df)):
        if df["Age"][i] <= 20:
            Age.append(0)
        elif df["Age"][i] <= 40:
            Age.append(1)
        else:
            Age.append(2)
    df["Age"] = Age
    # dispersed balance: 0-'low' 1-'middle' 2-'high'
    for i in range(len(df)):
        if df["Balance"][i] < 48000:
            Balance.append(0)
        elif df["Balance"][i] < 97198:
            Balance.append(1)
        else:
            Balance.append(2)
    df["Balance"] = Balance

    # dispersed EstimatedSalary: 0-'low' 1-'middle' 2-'high'
    for i in range(len(df)):
        if df["EstimatedSalary"][i] < 51002:
            EstimatedSalary.append(0)
        elif df["EstimatedSalary"][i] < 149388:
            EstimatedSalary.append(1)
        else:
            EstimatedSalary.append(2)
    df["EstimatedSalary"] = EstimatedSalary
    df.to_csv(outputPath)


discretization("Churn-Modelling-newT.csv", "Churn-Modelling-new-tree.csv")
# 3、数据筛选
# 由于原数据集中训练数据类别不均衡，为了达到较好的模型效果，
# 一般有过采样与欠采样两种方法可以解决类别不均衡问题，
# 这里采用了最简单的欠采样方法，将多余的类别数据删掉。

import pandas as pd

import pandas as pd


def filtering(dataPath, outputPath):
    df = pd.read_csv(dataPath)
    df_new = pd.DataFrame(
        columns=['Geography', 'EB', 'Age', 'EstimatedSalary', 'NumOfProducts', 'CreditScore', 'Tenure', 'HasCrCard',
                 'IsActiveMember', 'Exited', 'Gender'])
    ones = sum(df["Exited"])  # 统计标签
    length = len(df["EstimatedSalary"])
    zeros = length - ones
    i = 0
    flag_0 = 0
    flag_1 = 0

    while i != length:
        if df["Exited"][i] == 0 and flag_1 < 1 * ones:
            new_row = pd.DataFrame(
                {'Gender': [df["Gender"][i]], 'Geography': [df["Geography"][i]], 'EB': [df["EB"][i]],
                 'Age': [df["Age"][i]],
                 'EstimatedSalary': [df["EstimatedSalary"][i]], 'NumOfProducts': [df["NumOfProducts"][i]],
                 'CreditScore': [df["CreditScore"][i]], 'Tenure': [df["Tenure"][i]], 'HasCrCard': [df["HasCrCard"][i]],
                 'IsActiveMember': [df["IsActiveMember"][i]], 'Exited': [df["Exited"][i]]}, index=[i])
            df_new = pd.concat([df_new, new_row], ignore_index=True)
            flag_1 += 1

        if df["Exited"][i] == 1 and flag_0 < 1 * zeros:
            new_row = pd.DataFrame(
                {'Gender': [df["Gender"][i]], 'Geography': [df["Geography"][i]], 'EB': [df["EB"][i]],
                 'Age': [df["Age"][i]],
                 'EstimatedSalary': [df["EstimatedSalary"][i]], 'NumOfProducts': [df["NumOfProducts"][i]],
                 'CreditScore': [df["CreditScore"][i]], 'Tenure': [df["Tenure"][i]], 'HasCrCard': [df["HasCrCard"][i]],
                 'IsActiveMember': [df["IsActiveMember"][i]], 'Exited': [df["Exited"][i]]}, index=[i])
            df_new = pd.concat([df_new, new_row], ignore_index=True)
            flag_0 += 1

        i += 1

    df_new.to_csv(outputPath, index=False)


# 使用示例
filtering("Churn-Modelling-new-tree.csv", "final.csv")

filtering("Churn-Modelling-new-tree.csv", "final.csv")

# 4、划分训练集及测试集
import pandas as pd
import numpy as np

csv = pd.read_csv("final.csv")
print(csv.shape)
csv_array = np.array(csv)
# 标签数据为array第11列 Exited
target = csv_array[:, 10]
# 第1列为编号，对决策树模型无意义，去除，将剩余列作为特征项
feature = csv_array[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]]
# 将数据集按4:1分为训练集和测试集
from sklearn.model_selection import train_test_split

feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.2,
                                                                          random_state=10)

# 设置决策树参数
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(criterion="gini", max_depth=6, min_samples_split=200)

dt_model.fit(feature_train, target_train)
scores = dt_model.score(feature_test, target_test)

predict_results = dt_model.predict(feature_test)  # 测试集根据决策树的预测结果
# 生成决策树模型图
# from graphviz import Source
# from sklearn.tree import export_graphviz
# import os

# image_path = "./images/decision_trees"
# os.makedirs(image_path, exist_ok=True)

# export_graphviz(dt_model,
#                 out_file=os.path.join(image_path,"bank_tree.dot"),
#                 feature_names=["Geography","EB","Age","EstimatedSalary","NumOfProducts","CreditScore","Tenure","HasCrCard","IsActiveMember","Gender"],
#                 class_names=["not exited","exited"],
#                 rounded=True,
#                 filled=True)
# s=Source.from_file(os.path.join(image_path,"bank_tree.dot"))
# s.view()
# 模型校验评估
# 绘制ROC曲线
from sklearn.metrics import roc_curve  # 导入ROC曲线函数
import matplotlib.pyplot as plt  # 导入作图库

fpr, tpr, thresholds = roc_curve(target_test, predict_results, pos_label=1)
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, linewidth=2, label='ROC curve')  # 作出ROC曲线
plt.plot([0, 1], [0, 1], 'k--', label='guess')
plt.title("ROC Curve", fontsize=25)
plt.xlabel('False Positive Rate', fontsize=20)  # 坐标轴标签
plt.ylabel('True Positive Rate', fontsize=20)  # 坐标轴标签
plt.ylim(0, 1.05)  # 边界范围
plt.xlim(0, 1.05)  # 边界范围
plt.legend(loc=4, fontsize=20)  # 图例
plt.show()  # 显示作图结果

# 混淆矩阵
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数
import sklearn.metrics as sm

cm = confusion_matrix(target_test, predict_results)  # 混淆矩阵
print(sm.classification_report(target_test, predict_results))
import matplotlib.pyplot as plt  # 导入作图库
# plt.figure(figsize=(10, 10))
# plt.matshow(cm, fignum=0,cmap=plt.cm.Blues)
# plt.colorbar() #颜色标签
# for x in range(len(cm)): #数据标签
#     for y in range(len(cm)):
#         plt.annotate(cm[x,y], xy=(x, y),fontsize=30, horizontalalignment='center', verticalalignment='center')

# plt.ylabel('Hypothesized class',fontsize=20) #坐标轴标签
# plt.xlabel('True class',fontsize=20) #坐标轴标签
# plt.show()

from sklearn.metrics import confusion_matrix

confusion_matrix(target_test, predict_results)

# 十折交叉验证
# from sklearn.model_selection import StratifiedKFold
# skfold = StratifiedKFold(n_splits=10,shuffle=False)
# x_axis=[] ; y_axis=[]
# k=0;max=0;min=100;sum=0
# for train_index,test_index in skfold.split(feature,target):
#     k+=1
#     skfold_feature_train=feature[train_index]
#     skfold_feature_test=feature[test_index]
#     skfold_target_train=target[train_index]
#     skfold_target_test=target[test_index]
#     dt_model.fit(skfold_feature_train,skfold_target_train)
#     scores = dt_model.score(skfold_feature_test,skfold_target_test)
#     x_axis.append(k)
#     y_axis.append(scores)
#     if scores>max:
#         max=scores
#     if scores<min:
#         min=scores
#     sum+=scores
# avg=sum/k

# import matplotlib.pyplot as plt
# plt.plot(x_axis,y_axis)
# plt.ylim(0.6,0.9)
# plt.xlim(1,10)
# plt.xlabel("Rounds")
# plt.ylabel('True Rate')
# plt.title("KFold Cross Validation (k=%s) avg=%s"%(k,round(avg*100,2))+"%"+" max:"+"%s"%(round(max*100,2))+"%"+" min:"+"%s"%(round(min*100,2))+"%")
# plt.show()


# # 使用SVM进行分类
from sklearn import svm

clf = svm.SVC()
clf.fit(feature_train, target_train)
clf.predict(feature_test)
scores_svm = clf.score(feature_test, target_test)

from sklearn.metrics import confusion_matrix  # 导入混淆矩阵函数

cm = confusion_matrix(target_test, clf.predict(feature_test))  # 混淆矩阵
print(sm.classification_report(target_test, clf.predict(feature_test)))
# import matplotlib.pyplot as plt #导入作图库
# plt.figure(figsize=(10, 10))
# plt.matshow(cm, fignum=0,cmap=plt.cm.Blues)
# plt.colorbar() #颜色标签
# for x in range(len(cm)): #数据标签
#     for y in range(len(cm)):
#         plt.annotate(cm[x,y], xy=(x, y),fontsize=30, horizontalalignment='center', verticalalignment='center')

# plt.ylabel('Hypothesized class',fontsize=20) #坐标轴标签
# plt.xlabel('True class',fontsize=20) #坐标轴标签
# plt.show()

# # 使用神经网络进行分类
# from sklearn.neural_network import MLPClassifier
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(10, 11), random_state=1)
# mlp.fit(feature_train,target_train)
# mlp.predict(feature_test)
# scores_mlp = mlp.score(feature_test,target_test)

# from sklearn.metrics import confusion_matrix #导入混淆矩阵函数
# cm = confusion_matrix(target_test, mlp.predict(feature_test)) #混淆矩阵
# import matplotlib.pyplot as plt #导入作图库
# plt.figure(figsize=(10, 10))
# plt.matshow(cm, fignum=0,cmap=plt.cm.Blues)
# plt.colorbar() #颜色标签
# for x in range(len(cm)): #数据标签
#     for y in range(len(cm)):
#         plt.annotate(cm[x,y], xy=(x, y),fontsize=30, horizontalalignment='center', verticalalignment='center')

# plt.ylabel('Hypothesized class',fontsize=20) #坐标轴标签
# plt.xlabel('True class',fontsize=20) #坐标轴标签
# plt.show()
