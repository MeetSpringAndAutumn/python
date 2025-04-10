import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据集
data = pd.read_csv('wholesale.csv')

# 归一化处理
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)
normalized_df = pd.DataFrame(normalized_data, columns=data.columns)

# 查看相关系数
correlation_matrix = normalized_df.corr()

# 使用热力图可视化相关系数矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()
# 使用PCA进行降维
pca = PCA()
pca.fit(normalized_df)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance_ratio = explained_variance_ratio.cumsum()
n_components = 2  # 这里选择2个主成分进行演示
pca = PCA(n_components=n_components)
pca.fit(normalized_df)
pca_transformed = pca.transform(normalized_df)

# 使用K-means对降维后的数据进行聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(pca_transformed)
cluster_labels = kmeans.labels_

# 可视化聚类结果
plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1], c=cluster_labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Clustered Data Visualization')
plt.colorbar(label='Cluster Label')
plt.show()
