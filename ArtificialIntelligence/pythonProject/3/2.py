import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 读取数据
with open('car.txt', 'r') as file:
    lines = file.readlines()

# 提取特征和标签
data = []
labels = []
for line in lines:
    line = line.strip().split(',')
    data.append(line[:-1])  # 提取特征
    labels.append(line[-1])  # 提取标签

data = np.array(data)
labels = np.array(labels)

# 对特征进行 LabelEncoder 编码和 OneHotEncoder 编码
label_encoders = []
data_encoded1 = None
for i in range(data.shape[1]):
    label_encoder = LabelEncoder()
    data_encoded_column = label_encoder.fit_transform(data[:, i])
    if data_encoded1 is None:
        data_encoded1 = data_encoded_column.reshape(-1, 1)
    else:
        data_encoded1 = np.hstack((data_encoded1, data_encoded_column.reshape(-1, 1)))
    label_encoders.append(label_encoder)

onehot_encoder = OneHotEncoder()
data_encoded2 = onehot_encoder.fit_transform(data_encoded1)

# 对标签进行 LabelEncoder 编码
label_encoder_label = LabelEncoder()
labels_encoded = label_encoder_label.fit_transform(labels)

# 输出编码后的数据
print("编码后的特征数据（使用LabelEncoder编码后的数据）：")
print(data_encoded1)

print("编码后的特征数据（使用OneHotEncoder编码后的数据）：")
print(data_encoded2.toarray())

print("\n编码后的标签数据：")
print(labels_encoded)

# 反向解码示例
sample_index = 0  # 假设要解码的是第一个样本
decoded_label = label_encoder_label.inverse_transform([labels_encoded[sample_index]])[0]
print("\n反向解码的标签：")
print(decoded_label)
