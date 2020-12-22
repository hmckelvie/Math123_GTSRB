import numpy as np
import os
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

NUM_CLUSTER = 7 #there are seven classes of signs

'''
gmm
param: img_features, the features of the images after PCA with 50 components
returns: labels assigned through the gmm and the sihouette score for the clustering
computes the gaussian mixure model labels and computes the silhouette score of those
assignments
'''
def gmm(img_features):
    gmm = GaussianMixture(n_components=NUM_CLUSTER, covariance_type='full')
    gmm = gmm.fit(img_features)
    labels = gmm.predict(img_features)
    silhouette = silhouette_score(img_features, labels)
    return labels, silhouette
