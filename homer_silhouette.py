from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
from kmodes import kmodes, kprototypes
import pdb

def cluster_and_plot(X, n_clusters):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    ax.set_xlim([-0.6, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters; can't set random seed with kmodes
    km = kmodes.KModes(n_clusters=n_clusters, init='Huang', n_init=5, max_iter=5, verbose=2)
    cluster_labels = km.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed clusters
    silhouette_avg = silhouette_score(X, cluster_labels, metric='hamming')

    print('For n_clusters = {} the average silhouette_score is: {}.'format(n_clusters, silhouette_avg))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels, metric='hamming')

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title('The silhouette plot for the various clusters')
    ax.set_xlabel('The silhouette coefficient values')
    ax.set_ylabel('Cluster label')

    # The vertical line for average silhoutte score of all the values
    ax.axvline(x=silhouette_avg, color='red', linestyle='--')

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.title('Silhouette analysis for KModes with {} clusters'.format(n_clusters))

    plt.savefig('img/silhouette_{}n.png'.format(n_clusters), dpi=200)
    plt.close()

def get_silhouette_score(X, n_clusters):
    km = kmodes.KModes(n_clusters=n_clusters, init='Huang', n_init=5, max_iter=5, verbose=2)
    cluster_labels = km.fit_predict(X)
    sil_avg = silhouette_score(X, cluster_labels, metric='hamming')
    return sil_avg

def plot_sil_scores(X):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    sil_scores = [get_silhouette_score(X, i) for i in range(2,9)]
    ax.plot(range(2,9), sil_scores)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')

    plt.savefig('img/silhouette_v_clusters.png', dpi=200)
    plt.close()
