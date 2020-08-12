import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

variables = pandas.read_csv('Dataset2.csv')
Y = variables[['X']]
X = variables[['Y']]
# print(X)
# print(Y)
index = [i for i in range(1, 20)]
kmeans = [KMeans(n_clusters=i) for i in range(1, 20)]
# print(kmeans)
index = [i for i in range(1, 20)]
kmeans = [KMeans(n_clusters=i) for i in range(1, 20)]
# print(kmeans)
score = [-kmeans[i].fit(variables).score(variables) for i in range(len(kmeans))]
print(score)
plt.plot(index, score)
plt.xticks(index, rotation="vertical")
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

pca = PCA(n_components=1).fit(variables)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)

kmeans=KMeans(n_clusters=6).fit(variables)
kmeansoutput=kmeans
print(kmeansoutput)
plt.figure('6 Cluster K-Means')
plt.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('6 Cluster K-Means')
plt.show()