import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from kmodes import kmodes, kprototypes
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from homer_cluster import one_hot, prep_kmodes
import pdb
import boto3
import os


def plot_silhouette(df, X, n_clusters, model='KM'):
    '''
    df: must be dense array or pandas dataframe
    '''
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    ax.set_xlim([-0.6, 1])
    # Insert blank space between silhouette plots of individual clusters
    ax.set_ylim([0, len(df) + (n_clusters + 1) * 10])

    # Initialize clusterer and set random state, if possible
    if model == 'AG':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average').fit(X)
        labels = clusterer.labels_

    elif model == 'KM':
        clusterer = kmodes.KModes(n_clusters=n_clusters, n_init=3, init='Huang', verbose=1)
        labels = clusterer.fit_predict(df)

    elif model == 'GM':
        clusterer = GaussianMixture(n_components=n_clusters, covariance_type='tied', max_iter=20, n_init=50, random_state=42, verbose=1).fit(X)
        labels = clusterer.predict(X)

    # Compute the silhouette score (average value for all the samples) and the silhoutte score for each sample
    silhouette_avg = silhouette_score(X, labels, metric='hamming')
    sample_silhouette_values = silhouette_samples(X, labels, metric='hamming')

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[labels == i]

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

    # Add a vertical line for average silhoutte score of all values
    ax.axvline(x=silhouette_avg, color='red', linestyle='--')

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.title('Silhouette analysis for {} with {} clusters'.format(clusterer.__class__.__name__, n_clusters))

    plt.savefig('sil_{}_{}.png'.format(clusterer.__class__.__name__, n_clusters), dpi=200)

    plt.close()

def get_silhouette_score(df, X, n_clusters, model='AG'):
    '''
    X: must be dense array or pandas dataframe
    '''
    # Initialize clusterer and set random state, if possible
    if model == 'AG':
        clusterer = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='average').fit(X)
        labels = clusterer.labels_
        sil_avg = silhouette_score(X, labels, metric='hamming')

    elif model == 'KM':
        clusterer = kmodes.KModes(n_clusters=n_clusters, n_init=5, init='Huang', verbose=1)
        labels = clusterer.fit_predict(df)
        sil_avg = silhouette_score(X, labels, metric='hamming')

    elif model == 'GM':
        clusterer = GaussianMixture(n_components=n_clusters, covariance_type='tied', max_iter=20, n_init=50, random_state=42, verbose=1).fit(X)
        labels = clusterer.predict(X)
        sil_avg = silhouette_score(X, labels, metric='hamming')

    return sil_avg

def plot_sil_scores(df, X, model):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    sil_scores = [get_silhouette_score(df, X, i, model) for i in range(3,9)]
    ax.plot(range(3,9), sil_scores)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')

    plt.savefig('sil_v_clust_{}.png'.format(model), dpi=200)
    plt.close()

def to_bucket(f, bucket, write_name):
    '''
    Write files to s3 bucket.
    INPUT: f - file to write
           bucket - bucket to write to
           write_name - name for S3
    '''
    # Specify the service
    s3 = boto3.resource('s3')
    data = open(f, 'rb')
    s3.Bucket(bucket).put_object(Key=write_name, Body=data)
    print('Success! {} added to {} bucket'.format(write_name, bucket))

if __name__ == '__main__':
    plt.close('all')
    df = pd.read_pickle('s3a://capstone-sking/df.pkl')
    bucket = 'capstone-sking'

    # df = pd.read_pickle('data/df.pkl')
    # df_users = pd.read_pickle('data/df_users.pkl')

    # df_users_km = prep_kmodes(df_users)
    df_km = prep_kmodes(df)


    X = one_hot(df)
    # X_users = one_hot(df_users)

    for i in range(2, 9):
        plt = plot_silhouette(df_km, X.todense(), n_clusters=i, model='KM')
        # cluster_and_plot(X_users.toarray(), n_clusters=i, model='AG')

    plot_sil_scores(df_km, X.todense(), model='KM')
    # plot_sil_scores(X_users.toarray(), model='AG')

    # for plot in plots:
    #     to_bucket(f=, bucket=bucket, write_name='sil_plot.csv')
