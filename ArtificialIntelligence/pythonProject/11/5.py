import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def train_and_evaluate(data, connectivity=None):
    """
    训练 AGNES 模型并计算轮廓系数得分
    """
    # 训练 AGNES 模型
    AgglomerativeClustering()
    agnes = AgglomerativeClustering(n_clusters=5, connectivity=connectivity)
    labels = agnes.fit_predict(data)

    # 计算轮廓系数得分
    silhouette_avg = silhouette_score(data, labels)

    # 绘制聚类结果
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title('Agglomerative Clustering Result')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

    return silhouette_avg


# 生成三种不同形状的数据
from sklearn.cluster import AgglomerativeClustering

def add_noise(x, y, amplitude):
    X = np.concatenate((x, y))
    X += amplitude * np.random.randn(2, X.shape[1])
    return X.T
def get_spiral(t, noise_amplitude=0.5):
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return add_noise(x, y, noise_amplitude)
def get_rose(t, noise_amplitude=0.02):
    # Equation for "rose" (or rhodonea curve); if k is odd, then
    # the curve will have k petals, else it will have 2k petals
    k = 5
    r = np.cos(k*t) + 0.25
    x = r * np.cos(t)
    y = r * np.sin(t)
    return add_noise(x, y, noise_amplitude)
def get_hypotrochoid(t, noise_amplitude=0):
    a, b, h = 10.0, 2.0, 4.0
    x = (a - b) * np.cos(t) + h * np.cos((a - b) / b * t)
    y = (a - b) * np.sin(t) - h * np.sin((a - b) / b * t)
    return add_noise(x, y, 0)


n_samples=500
np.random.seed(2)
t = 2.5 * np.pi * (1 + 2 * np.random.rand(1, n_samples))
# X = get_spiral(t)
X = get_rose(t)
# X = get_hypotrochoid(t)

silhouette_score = train_and_evaluate(X)

# 输出轮廓系数得分
print("Silhouette Score for  Data:", silhouette_score)
