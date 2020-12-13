from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''
analyze_labels
param: labeled_images, dataframe that includes each image and its expected label
       gmm_label, dataframe with each image and the label predicted with gmm
       kmeans_label, dataframe with each image and the label predicted with kmeans
returns: none, writes to files
computes the voted label with each clustering method and produces visuals of each
cluster assignment
'''
def analyze_labels(labeled_images, gmm_labels, kmeans_labels):
    gmm_assignment = pd.DataFrame(gmm_labels, columns=['gmm_label'])
    kmeans_assignment = pd.DataFrame(kmeans_labels, columns=['kmeans_label'])

    labeled = labeled_images.join(gmm_assignment)
    labeled = labeled.join(kmeans_assignment)
    labeled.to_csv('expected_unsupervised_labels.csv', index=False)

    voted_label = pd.DataFrame(columns=['folder','gmm_label','kmeans_label'])
    for i in range(NUM_CLUSTER):
        curr_label = labeled.loc[labeled['expected_label'] == i]
        assigned_gmm = labeled.loc[labeled['gmm_label'] == i]
        assigned_kmeans = labeled.loc[labeled['kmeans_label'] == i]

        curr_label.to_csv(('expected_cluster/clustering_sign_%d.csv'%(i)), index=False)
        assigned_gmm.to_csv(('gmm_cluster/gmm_cluster_%d.csv'%(i)), index=False)
        assigned_kmeans.to_csv(('kmeans_cluster/kmeans_cluster_%d.csv'%(i)), index=False)

        curr_vote = get_votes(curr_label)
        voted_label = voted_label.append(curr_vote, ignore_index=True)
        
    voted_label.to_csv('voted_label_per_method.csv', index=False)
    
'''
get_votes
param: curr_label, the assignments for each actual label in the dataset
returns: none
computes the majority voted label from each method for the respective sign
'''
def get_votes(curr_label):
    gmm_vote = curr_label['gmm_label'].mode()
    gmm_vote = gmm_vote.to_numpy()

    kmeans_vote = curr_label['kmeans_label'].mode()
    kmeans_vote = kmeans_vote.to_numpy()
    curr_vote = pd.DataFrame({'folder': [int(i)], 'gmm_label': [gmm_vote[0]], 'kmeans_label': [kmeans_vote[0]]})

    return curr_vote

'''
graph_all_clusters
param: none
return: none, writes to files
For each assigned gmm and kmeans cluster produces bar graphs showing which images are
present in the cluster.
'''
def graph_all_clusters() :
    pp_gmm = PdfPages('gmm_cluster_graph.pdf')
    pp_kmeans = PdfPages('kmeans_cluster_graph.pdf')
    for i in range(NUM_CLUSTER):
      cluster_graph(('gmm_cluster/gmm_cluster_%d.csv'%(i)), i, pp_gmm, "gmm")
      cluster_graph(('kmeans_cluster/kmeans_cluster_%d.csv'%(i)), i, pp_kmeans, "kmeans")
    pp_gmm.close()
    pp_kmeans.close()

'''
cluster_graph
param: filename, the file containing assignments for that cluster
       cluster_num, the id of that cluster given by the clustering method
       pp, the pdf to write the graph to
       cluster_type, the clustering method
returns: none, writes graphs to file
forms bar graph for the respective cluster, showing the counts of each sign type in the
cluster
'''
def cluster_graph(filename, cluster_num, pp, cluster_type):
    cluster = pd.read_csv(filename)
    del cluster['gmm_label']
    del cluster['kmeans_label']
    counts = cluster['expected_label'].value_counts()
    ax = counts.plot.bar(title=("%s cluster %d"%(cluster_type, cluster_num)))
    plt = ax.get_figure()
    plt.savefig(pp, format='pdf')

