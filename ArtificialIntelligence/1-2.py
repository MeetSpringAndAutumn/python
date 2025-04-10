import pandas as pd

# 读取数据
data = pd.read_csv(
    "D:\\Desktop\\python\\ArtificialIntelligence\\stock_data.csv",
    encoding="gbk")

# (1) 提取数值部分的数据，输出数据的维度
numerical_data = data.select_dtypes(include=["number"])
print("数据维度：", numerical_data.shape)

# (2) 对每一列求均值、最大值、最小值并输出
for column in numerical_data.columns:
    mean = numerical_data[column].mean()
    max_value = numerical_data[column].max()
    min_value = numerical_data[column].min()
    print(f"列：{column}")
    print(f"均值：{mean}")
    print(f"最大值：{max_value}")
    print(f"最小值：{min_value}\n")

# (3) 选择其中一列作为排序依据对整个数据进行排序
sort_column = numerical_data.columns[0]
sorted_data = numerical_data.sort_values(by=sort_column)
print("排序后的数据：", sorted_data)

# (4) 删除最后一列，对删除后的数据进行归一化处理
normalized_data = sorted_data.drop(columns=sorted_data.columns[-1])
for column in normalized_data.columns:
    max_value = normalized_data[column].max()
    min_value = normalized_data[column].min()
    normalized_data[column] = (
        normalized_data[column] - min_value) / (max_value - min_value)
print("归一化后的数据：", normalized_data)
