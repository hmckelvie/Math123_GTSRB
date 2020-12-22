import pandas as pd
import numpy as np
import os

'''
assigned_images
param: lalbed_images, data fram of labeled images, 
       gmm_labels, labels determined with gmm
       kmeans_labels, labels determined with kmeans
       seg_num, the segmentation number, zero if none
returns dataframe containing all labels, also writes to file
'''
def assigned_images(labeled_images, gmm_labels, kmeans_labels, seg_num):
    gmm_assignment = pd.DataFrame(gmm_labels, columns=['gmm_label'])
    kmeans_assignment = pd.DataFrame(kmeans_labels, columns=['kmeans_label'])

    labeled = labeled_images.join(gmm_assignment)
    labeled = labeled.join(kmeans_assignment)
    if seg_num != 0:
        labeled.to_csv(('expected_unsupervised_labels_seg_%d.csv'%(seg_num)), index=False)
    else:
        labeled.to_csv('expected_unsupervised_labels.csv', index=False)
    return labeled 

'''
output_files
param: labeled, the dataframed of all labels for each image
       segmented, the segmentation number, zero if no segmentation
return: none, writes to files
writes the cluster information to each respective directery
'''
def output_files(labeled, segmented):
    for i in range(NUM_CLUSTER):
        curr_label =  labeled.loc[labeled['expected_cluster'] == i]
        assigned_gmm = labeled.loc[labeled['gmm_label'] == i]
        assigned_kmeans = labeled.loc[labeled['kmeans_label'] == i]

        if segmented != 0: 
            if not os.path.isdir('expected_cluster_seg_%d/'%(segmented)) :
                os.mkdir('expected_cluster_seg_%d'%(segmented)) 
            if not os.path.isdir('gmm_cluster_seg_%d/'%(segmented)):
                os.mkdir('gmm_cluster_seg_%d'%(segmented))
            if not os.path.isdir('kmeans_cluster_seg_%d/'%(segmented)):
                os.mkdir('kmeans_cluster_seg_%d'%(segmented))

            curr_label.to_csv(('expected_cluster_seg_%d/clustering_sign_%d.csv'%(segmented, i)), index=False)
            assigned_gmm.to_csv(('gmm_cluster_seg_%d/gmm_cluster_%d.csv'%(segmented, i)), index=False)
            assigned_kmeans.to_csv(('kmeans_cluster_seg_%d/kmeans_cluster_%d.csv'%(segmented, i)), index=False)
        else: 
            if not os.path.isdir('expected_cluster/') :
                os.mkdir('expected_cluster') 
            if not os.path.isdir('gmm_cluster/'):
                os.mkdir('gmm_cluster')
            if not os.path.isdir('kmeans_cluster/'):
                os.mkdir('kmeans_cluster')

            curr_label.to_csv(('expected_cluster/clustering_sign_%d.csv'%(i)), index=False)
            assigned_gmm.to_csv(('gmm_cluster/gmm_cluster_%d.csv'%(i)), index=False)
            assigned_kmeans.to_csv(('kmeans_cluster/kmeans_cluster_%d.csv'%(i)), index=False)

'''
match_labels
param: labeled, dataframe of the labels for each image
       cluster_counts, the number of images in each given category as a datafram
       segmented, the segmentation number, zero if no segmentation
returns: the dataframe of the voted label for each category and each clustering method
matches the clusters from each method to a category using scaled majority voting
'''
def match_labels(labeled, cluster_counts, segmented):
    voted_label = pd.DataFrame(columns=['cluster','gmm_label','kmeans_label'])
    already_assigned_gmm = []
    already_assigned_kmeans = []
    for i in range(NUM_CLUSTER):
        assigned_gmm = labeled.loc[labeled['gmm_label'] == i]
        assigned_kmeans = labeled.loc[labeled['kmeans_label'] == i]
        gmm_vote = get_votes(assigned_gmm, cluster_counts, i, already_assigned_gmm)
        kmeans_vote = get_votes(assigned_kmeans, cluster_counts, i, already_assigned_kmeans)
        already_assigned_gmm.append(gmm_vote)
        already_assigned_kmeans.append(kmeans_vote)
        curr_vote = pd.DataFrame({'cluster': [int(i)], 'gmm_label': [gmm_vote], 'kmeans_label': [kmeans_vote]})
        voted_label = voted_label.append(curr_vote, ignore_index=True)
    if segmented != 0:
        voted_label.to_csv(('voted_label_per_method_seg_%d.csv'%(segmented)), index=False)
    else:
        voted_label.to_csv('voted_label_per_method.csv', index=False)
    return voted_label

'''
counts_per_cluster
param: labeled_images, dataframe of all labels for each image
returns: a numpy array of the count of images in each category
computes the number of images in each category
'''
def counts_per_cluster(labeled_images):
    counts = []
    for i in range(NUM_CLUSTER) : 
        curr_cluster = labeled_images.loc[labeled_images['expected_cluster'] == i]
        counts.append(len(curr_cluster))
    counts = np.array(counts)
    return counts

'''
get_votes
param: curr_label, the label assignment as a dataframe
       cluster_counts, the number of images in each cluster
       already_assigned, array containing which categories have already been matched
returns: the vote for that respective cluster
computes the category assignment for the cluster given
'''
def get_votes(curr_label, cluster_counts, i, already_assigned):
    max_ratio = 0
    max_cluster = 6
    for j in range(NUM_CLUSTER -1, -1, -1):
        current_cluster = curr_label.loc[curr_label['expected_cluster'] == (j)]
        curr_count = len(current_cluster.index)
        total = cluster_counts[(j)]
        ratio = float(curr_count) / float(total)
        if ratio > max_ratio and already_assigned.count(j) == 0:
            max_ratio = ratio
            max_cluster = (j)
    return max_cluster
'''
get_accuracy_rate
param: labeled, dataframe of all labeling 
       voted_cluster: dataframe of the voted clusters corollating to categories
returns: the calculated gmm and kmeans accuracy rate
computes the gmm and kmeans accuracy rate through counting each image that does not
get maps to the majority voted cluster for that category. 
'''
def get_accuracy_rate(labeled, voted_cluster):
    total = len(labeled.index)
    miss_gmm = 0
    miss_kmeans = 0
    for i in range(NUM_CLUSTER) :
        voted_label = voted_cluster.loc[voted_cluster['cluster'] == i]
        
        gmm_vote = voted_label['gmm_label']
        clustered_in_vote_gmm= labeled.loc[labeled['gmm_label'] == int(gmm_vote)]
        incorrect_gmm = clustered_in_vote_gmm.loc[clustered_in_vote_gmm['expected_cluster'] != i]
        miss_gmm = miss_gmm + len(incorrect_gmm.index)

        kmeans_vote = voted_label['kmeans_label']
        clustered_in_vote_kmeans= labeled.loc[labeled['kmeans_label'] == int(kmeans_vote)]
        incorrect_kmeans = clustered_in_vote_kmeans.loc[clustered_in_vote_kmeans['expected_cluster'] != i]
        miss_kmeans = miss_kmeans+ len(incorrect_kmeans.index)


    gmm_rate = (float(total) - float(miss_gmm)) / float(total)
    kmeans_rate = (float(total) - float(miss_kmeans)) / float(total)
    return gmm_rate, kmeans_rate
