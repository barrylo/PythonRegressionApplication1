from sklearn.datasets import make_blobs 
import matplotlib.pyplot as plt 
from itertools import product 
import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans 
  
  
X, y = make_blobs(n_samples=42, n_features=43, centers=3) 
print(X) 
  
  
columns = ['feature' + str(x) for x in np.arange(1, 44, 1)] 
d = {key: values for key, values in zip(columns, X.T)} 
d['label'] = y 
data = pd.DataFrame(d) 
  
data.to_csv('C:\PYDEV\ccc.csv') 
  
data = pd.read_csv('C:\PYDEV\dddd.csv') 
print(data) 
  
z=data.ix[:, :-1].values 
  
kmeans = KMeans(n_clusters=3) 
kmeans.fit_predict(z) 
colors = ["b.","r.","g.", "y."] 
  
centroids = kmeans.cluster_centers_ 
labels = kmeans.labels_ 
  
for i in range(len(z)): 
    #print("coordinate:",z, "label:", labels[i]) 
    plt.plot(z[i][0], z[i][1], colors[labels[i]], markersize = 10) 
  
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10) 
  
plt.show() 
