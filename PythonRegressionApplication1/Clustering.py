import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pandas as pd
from sklearn.cluster import KMeans


MY_FILE='C:\PYDEV\RES_1__intake.csv'
#MY_FILE='C:\PYDEV\total_watt_v2.csv'
date = []
consumption = []


df = pd.read_csv(MY_FILE, parse_dates=[0], index_col=[0])
df = df.resample('1D', how='count')
df = df.dropna()

date = df.index.tolist()
date = [x.strftime('%Y-%m-%d') for x in date]
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
date_numeric = encoder.fit_transform(date)
consumption = df[df.columns[0]].values


X = np.array([date_numeric, consumption]).T

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)

colors = ["b.","r.","g."]

for i in range(len(X)):
    print("coordinate:",X[i], "label:", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()

