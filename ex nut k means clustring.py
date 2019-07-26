import pandas as pd
import numpy as np

nut=pd.read_csv("D:/r python/data/nutrient.csv",index_col=0)

from sklearn.preprocessing import StandardScaler
# Create scaler: scaler
scaler = StandardScaler()
nutscaled=scaler.fit_transform(nut)
# Import KMeans
from sklearn.cluster import KMeans

# Create a KMeans instance with clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(nutscaled)
#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(nutscaled)

# Print cluster labels of new_points
print(labels)

clusterID = pd.DataFrame(labels)
clusteredData = pd.concat([nut.reset_index(),clusterID],axis=1)

# Variation
print(model.inertia_)

clustNos = [2,3,4,5,6,7,8,9,10]
Inertia = []

for i in clustNos :
    model = KMeans(n_clusters=i)
    model.fit(nutscaled)
    Inertia.append(model.inertia_)
    
# Import pyplot
import matplotlib.pyplot as plt

plt.plot(clustNos, Inertia, '-o')
plt.title("Scree Plot")
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(clustNos)
plt.show()

# Create a KMeans instance with clusters: model
model = KMeans(n_clusters=5)

# Fit model to points
model.fit(nutscaled)
#model.n_init
# Determine the cluster labels of new_points: labels
labels = model.predict(nutscaled)

clusterID = pd.DataFrame(labels)
clusteredData = pd.concat([nut.reset_index(drop=True),clusterID],axis=1)

## Scaling
# Perform the necessary imports
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler = StandardScaler()

# Create KMeans instance: kmeans
kmeans = KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline = make_pipeline(scaler,kmeans)

# Fit the pipeline to samples
pipeline.fit(nutscaled)

# Calculate the cluster labels: labels
labels = pipeline.predict(nutscaled)

# Display ct
print(labels)

