import pandas
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import matplotlib.colors as mc

variables = pandas.read_csv('Dataset2.csv')
Y = variables[['X']]
X = variables[['Y']]

db = DBSCAN(eps=2, min_samples=5).fit(variables)
labels =  db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
x = variables.values[:,0]
y = variables.values[:,1]
cmap = plt.cm.rainbow
norm = mc.BoundaryNorm(labels, cmap.N)
plt.figure(figsize=(14,7))
plt.scatter(x,y,c=labels,cmap='viridis',s=50)
plt.title('DBSCAN of Dataset2', fontsize = 20)
plt.show()