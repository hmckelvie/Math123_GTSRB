import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

NUM_CLUSTER = 43 #there are 43 types of signs

'''
kmeans
param: img_features, the features of the dataset, after PCA with 50 componenets is performed
returns: labels assigned through kmeans and the sihouette score
computes the kmeans clustering and computes the silhouette score of the clustering
'''
def kmeans(img_features):
    kmeans = KMeans(init='k-means++', n_clusters=NUM_CLUSTER, random_state=0)
    labels = kmeans.fit(img_features).predict(img_features)
    silhouette = silhouette_score(img_features, labels)
    return labels, silhouette

