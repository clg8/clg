from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
X,_=make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)
kmeans=KMeans(n_clusters=4,random_state=0)
kmeans.fit(X)
y_kmeans=kmeans.predict(X)
wcss=[]
for i in range(1,11):
    kmeans_final=KMeans(n_clusters=i,random_state=0)
    kmeans_final.fit(X)
    wcss.append(kmeans_final.inertia_)

plt.figure(figsize=(8,4))
plt.plot(range(1,11),wcss,marker='o')
plt.title("kmeans")
plt.xlabel("no of0 xluster")
plt.ylabel("wcss")
plt.show()
plt.figure(figsize=(8, 4))
plt.scatter(X[:,0],X[:,1],c=y_kmeans,s=50,cmap="viridis")
center=kmeans.cluster_centers_
plt.scatter(center[:,0],center[:,1],c="red",s=100,alpha=0.50)
plt.show()