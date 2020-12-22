import preprocess
import pca
import gmm
import kmeans
import accuracy_rate
import analyze_labels
import os

path = os.getcwd()
preprocess.traverse(path)
labeled_images = preprocess.load_preprocessed(path + "/Processed") 


img_features = pca.pca_features()
features_df = pd.DataFrame(img_features)
labeled_features = labeled_images.join(features_df)


gmm_labels, gmm_score = gmm.gmm(img_features)

kmeans_labels, kmeans_score = kmeans.kmeans(img_features)

analyze_unsupervised.analyze_labels(labeled_images, gmm_labels, kmeans_labels)
analyze_unsupervised.graph_all_clusters()
