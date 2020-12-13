import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

NUM_COMPONENTS= 50

'''
pca_features
param: none
returns: the features array with 50 components 
computes PCA with 50 componenets for the data in the features array
'''
def pca_features():
    with open(FEATURES_FILE, 'rb') as file:
        img_features = np.load(file)
        scaler = StandardScaler()    
        scaler.fit(img_features)
        img_features = scaler.transform(img_features)
        pca = PCA(n_components=NUM_COMPONENTS)  
        img_features = pca.fit_transform(img_features)
        return img_features   

