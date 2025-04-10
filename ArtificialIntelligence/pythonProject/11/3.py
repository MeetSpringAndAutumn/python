import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# 读取数据
data_multivar = np.loadtxt('data_multivar_cluster.txt', delimiter=',')
data_perf = np.loadtxt('data_perf.txt', delimiter=',')


def calculate_silhouette_scores(data, min_clusters=2, max_clusters=10):
    silhouette_scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, labels)
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores


min_clusters = 2
max_clusters = 10

silhouette_scores_multivar = calculate_silhouette_scores(data_multivar, min_clusters, max_clusters)
silhouette_scores_perf = calculate_silhouette_scores(data_perf, min_clusters, max_clusters)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.bar(range(min_clusters, max_clusters + 1), silhouette_scores_multivar)
plt.title('Silhouette Scores for data_multivar')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
print('Silhouette Scores for data_multivar')
print('Number of clusters : Silhouette Score')
for i in range(min_clusters, max_clusters + 1):
    print(f'{i}: {silhouette_scores_multivar[i-min_clusters]}')
plt.xticks(range(min_clusters, max_clusters + 1))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.bar(range(min_clusters, max_clusters + 1), silhouette_scores_perf)
plt.title('Silhouette Scores for data_perf')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
print('Silhouette Scores for data_perf')
print('Number of clusters : Silhouette Score')
for i in range(min_clusters, max_clusters + 1):
    print(f'{i}: {silhouette_scores_perf[i-min_clusters]}')
plt.xticks(range(min_clusters, max_clusters + 1))
plt.grid(True)

plt.tight_layout()
plt.show()
