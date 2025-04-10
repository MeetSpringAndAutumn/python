import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# 读取数据
data_perf = np.loadtxt('data_perf.txt', delimiter=',')

def calculate_silhouette_scores(data, epsilon_values):
    """
    计算不同距离参数值下的轮廓系数得分
    """
    silhouette_scores = []
    for epsilon in epsilon_values:
        dbscan = DBSCAN(eps=epsilon)
        labels = dbscan.fit_predict(data)
        if len(np.unique(labels)) > 1:  # 至少有两个类别才能计算轮廓系数
            silhouette_avg = silhouette_score(data, labels)
        else:
            silhouette_avg = -1  # 如果只有一个类别，则轮廓系数为-1
        silhouette_scores.append(silhouette_avg)
    return silhouette_scores

def find_best_epsilon(data, epsilon_values):
    """
    找到轮廓系数得分最大的距离参数值
    """
    best_epsilon = None
    best_score = -1
    best_model = None
    best_labels = None
    for epsilon in epsilon_values:
        dbscan = DBSCAN(eps=epsilon)
        labels = dbscan.fit_predict(data)
        if len(np.unique(labels)) > 1:  # 至少有两个类别才能计算轮廓系数
            silhouette_avg = silhouette_score(data, labels)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_epsilon = epsilon
                best_model = dbscan
                best_labels = labels
    return best_epsilon, best_model, best_labels

def check_unclustered_points(labels):
    """
    检查标签中是否有未聚类的点，并输出最终聚类的个数
    """
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        num_clusters = len(unique_labels) - 1
    else:
        num_clusters = len(unique_labels)
    return num_clusters


def plot_clusters(data, labels):
    """
    绘制聚类结果
    """
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels) - 1 if -1 in unique_labels else len(unique_labels)
    markers = ['o', 's', '^', 'x', '+', '*', 'p', 'h', 'd', '|']
    colors = plt.cm.get_cmap('tab10', num_clusters)
    for i, label in enumerate(unique_labels):
        if label == -1:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], marker='.', color='k', label='Noise')
        else:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], marker=markers[i], color=colors(i),
                        label=f'Cluster {label}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering Result')
    plt.legend()
    plt.grid(True)
    plt.show()

# 使用10个距离参数统计轮廓系数得分
epsilon_values = np.linspace(0.1, 2.0, 10)
silhouette_scores = calculate_silhouette_scores(data_perf, epsilon_values)

# 绘制轮廓系数得分条形图
plt.figure(figsize=(10, 6))
plt.bar(epsilon_values, silhouette_scores, width=0.1)
plt.xlabel('Epsilon')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Different Epsilon Values')
plt.grid(True)
plt.show()

# 返回轮廓系数得分最大的距离参数值，model，labels
best_epsilon, best_model, best_labels = find_best_epsilon(data_perf, epsilon_values)
# print(f"Best epsilon value: {best_epsilon}")

# 查找labels中是否有未聚类的点，输出最终聚类的个数
num_clusters = check_unclustered_points(best_labels)
print(f"Number of clusters: {num_clusters}")

# 分别不用不同的marker（根据类别个数）画出样本数据，未聚类的使用标记’.‘
plot_clusters(data_perf, best_labels)
