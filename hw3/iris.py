import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

iris = load_iris()
x = iris.data
y = iris.target
class_names = iris.target_names

x_scaled = StandardScaler().fit_transform(x)

k_nums = [i for i in range(2, 7)]

plt_x = []
plt_y = []

for i in k_nums:
    kmeans = KMeans(n_clusters = i, random_state = 0, n_init='auto').fit(x_scaled)
    intertia = kmeans.inertia_
    print(f'K={i}, Inertia={intertia}')
    plt_x.append(i)
    plt_y.append(intertia)

x = plt_x
y = plt_y

plt.plot(x, y)

plt.title("K Clusters vs Inertia (Iris)")
plt.xlabel("K Values")
plt.xticks(k_nums) 
plt.ylabel("Inertia")

plt.show()

x_pca = PCA(n_components=2).fit_transform(x_scaled)
kmeans = KMeans(n_clusters=3, random_state=0).fit(x_pca)

plt.scatter(x_pca[:,0], x_pca[:,1], c=kmeans.labels_, s=20, alpha=0.7)
plt.title("K Means Clustering (Iris)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=3, metric='euclidean')
y_pred = hierarchical.fit_predict(x_pca)

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_pred, s=20, alpha=0.7)
plt.title('Hierarchical Clustering (Iris)')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
