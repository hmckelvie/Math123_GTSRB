import preprocess
import image_segmentation
import pca
import gmm
import kmeans
import accuracy_rate
import analyze_labels
import os

FEATURES_FILE = 'preprocessed_features.npy'
FEATURES_SEG_FILE = 'preprocessed_segmented_features.npy'
SEG_NUM = 100

path = os.getcwd()
preprocess.traverse(path)
image_segmentation.preprocess_segmentation(path, SEG_NUM)

seg_path = path+("/Processed_Segmented_%d"%(SEG_NUM)) 

labeled_images = preprocess.load_preprocessed(path + "/Processed", FEATURES_FILE) 
labeled_images_seg = preprocess.load_preprocessed(seg_path, FEATURES_SEG_FILE) 

cluster_counts = accuracy_rate.counts_per_cluster(labeled_images)
cluster_counts_seg = accuracy_rate.counts_per_cluster(labeled_images_seg)

img_features = pca.pca_features(FEATURES_FILE)
features_df = pd.DataFrame(img_features)
labeled_features = labeled_images.join(features_df)

img_features_seg = pca.pca_features(FEATURES_SEG_FILE)
features_seg_df = pd.DataFrame(img_features_seg)
labeled_features_seg = labeled_images.join(features_seg_df)


gmm_labels, gmm_score = gmm.gmm(img_features)
gmm_labels_seg, gmm_score_seg = gmm.gmm(img_features_seg)

kmeans_labels, kmeans_score = kmeans.kmeans(img_features)
kmeans_labels_seg, kmeans_score_seg = kmeans.kmeans(img_features_seg)

labeled = accuracy_rate.assigned_images(labeled_images, gmm_labels, kmeans_labels, 0)
accuracy_rate.output_files(labeled, 0)
voted_labels = accuracy_rate.match_labels(labeled, cluster_counts, 0)

labeled_seg = accuracy_rate.assigned_images(labeled_images_seg, gmm_labels_seg, kmeans_labels_seg, SEG_NUM)
accuracy_rate.ouput_files(labeled_seg, SEG_NUM)
voted_labels_seg = accuracy_rate.match_labels(labeled_seg, cluster_counts_seg, SEG_NUM)

gmm_rate, kmeans_rate = get_accuracy_rate(labeled, voted_labels)
gmm_rate_seg, kmeans_rate_seg = get_accuracy_rate(labeled_seg, voted_labels_seg)

print("GMM ACCURACY: %f" %(gmm_rate))
print("KMEANS ACCURACY: %f"(kmeans_rate))
print("%d SEGMENTED KMEANS ACCURACY: %f"(SEG_NUM, kmeans_rate_seg))
print("%d SEGMENTED GMM ACCURACY: %f"(SEG_NUM, gmm_rate_seg))

analyze_labels.graph_all_clusters()
analyze_labels.graph_all_seg_clusters(SEG_NUM)
