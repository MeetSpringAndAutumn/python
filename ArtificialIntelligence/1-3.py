import pandas as pd
import os
import sys
os.chdir(sys.path[0])
# 读取数据
iris_data = pd.read_csv("iris.csv")

# (1) 将前四列(不含编号列)读取到data变量中，最后一列读取到labels变量中
data = iris_data.iloc[:, 1:5]
labels = iris_data.iloc[:, -1]

# (2) 输出labels的类别及每个类别出现的数量
print("类别及数量：")
print(labels.value_counts())

# (3) 输出labels值为setosa对应的样本
setosa_data = data[labels == "setosa"]
print("setosa样本：", setosa_data)

# (4) 筛选第一列值小于5.0，第三列值大于1.5的所有data
filtered_data = data[(data.iloc[:, 0] < 5.0) & (data.iloc[:, 2] > 1.5)]
print("筛选后的data：", filtered_data)
