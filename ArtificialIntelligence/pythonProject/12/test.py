import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as ms
from sklearn import cluster, covariance, manifold
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

df = pd.read_csv('Mall_Customers.csv')
sns.pairplot(df.iloc[:, 1:], diag_kind='kde')
plt.show()

# 绘制不同性别在年龄与年收入之间的关系  
plt.figure(figsize=(10, 5))
for gender in ['Male', 'Female']:
    plt.scatter(x='Age', y='Annual Income (k$)', data=df[df['Gender'] == gender],
                s=200, alpha=0.5, label=gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)')
plt.title('Age and Annual Income')
plt.legend()
plt.show()
# 不同性别在年龄与年收入之间的关系
plt.figure(figsize=(10, 5))
for gender in ['Male', 'Female']:
    plt.scatter(x='Age', y='Spending Score (1-100)', data=df[df['Gender'] == gender],
                s=200, alpha=0.5, label=gender)
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.title('Age and Spending Score')
plt.legend()
plt.show()
# 不同性别在年收入与消费得分之间的关系
plt.figure(1, figsize=(10, 5))
for gender in ['Male', 'Female']:
    plt.scatter(x='Annual Income (k$)', y='Spending Score (1-100)',
                data=df[df['Gender'] == gender], s=200, alpha=0.5, label=gender)
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Annual Income and Spending Score')
plt.legend()
plt.show()
X = df.iloc[:, 2:].values
inertia = []
score = []
for i in range(2, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)
    km.fit(X)
    labels = km.labels_
    score.append(ms.silhouette_score(X, labels))
    inertia.append(km.inertia_)
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(range(2, 11), inertia, 'o-')
plt.title('Inertia')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

plt.subplot(122)
plt.bar(range(2, 11), score)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.tight_layout()
plt.show()