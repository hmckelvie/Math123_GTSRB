from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
'''
graph_all_seg_clusters
param: the segmentation number
return: none, writes to files
For each assigned gmm and kmeans cluster produces bar graphs showing which images are
present in the cluster.
'''
def graph_all_seg_clusters(seg_num) :
    pp_gmm = PdfPages('seg_%d_gmm_cluster_graph.pdf'%(seg_num))
    pp_kmeans = PdfPages('seg_%d_kmeans_cluster_graph.pdf'%(seg_num))
    for i in range(NUM_CLUSTER):
      cluster_graph(('gmm_cluster_seg/gmm_cluster_%d.csv'%(i)), i, pp_gmm, "gmm", seg_num)
      cluster_graph(('kmeans_cluster_seg/kmeans_cluster_%d.csv'%(i)), i, pp_kmeans, "kmeans", seg_num)
    pp_gmm.close()
    pp_kmeans.close()

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

