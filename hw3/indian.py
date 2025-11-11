import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA

indianR = loadmat("indianR.mat")
indianGth = loadmat("indian_gth.mat")

x = indianR['X'].T 
y = indianGth['gth'].flatten()

# remove background class
mask = y > 0
x = x[mask]
y = y[mask] 

x_scaled = StandardScaler().fit_transform(x)


k_nums = [i for i in range(2, 17)]
print(k_nums)

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

plt.title("K Clusters vs Inertia (Indian Pines)")
plt.xlabel("K Values")
plt.xticks(k_nums) 
plt.ylabel("Inertia")

plt.show()

x_pca = PCA(n_components=2).fit_transform(x_scaled)
kmeans = KMeans(n_clusters=6).fit(x_pca)

plt.scatter(x_pca[:,0], x_pca[:,1], c=kmeans.labels_, s=6, alpha=0.7)
plt.title("K Means Clustering (Indian Pines)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=6, metric='euclidean')
y_pred = hierarchical.fit_predict(x_pca)

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=y_pred, s=6, alpha=0.7)
plt.title('Hierarchical Clustering (Indian Pines)')
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
