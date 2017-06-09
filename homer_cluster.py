import numpy as np
import pandas as pd
from kmodes import kmodes, kprototypes
from sklearn.preprocessing import scale, LabelEncoder, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mca
import pickle
import pdb
from homer_silhouette import cluster_and_plot, plot_sil_scores
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.neighbors import kneighbors_graph


def one_hot(df_):
    df = df_.copy()

    cols_to_keep = ['UserRole',
                    'OrganizationType',
                    'Sample',
                    'ImportedWind',
                    'ImportedSolar',
                    'GenCostMultiLines',
                    'WindCostMultiLines',
                    'BatCostMultiLines',
                    'PvCostMultiLines',
                    'ConCostMultiLines',
                    ]

    df = df[cols_to_keep]

    df['UserRole'] = df['UserRole'].astype(object)
    df['OrganizationType'] = df['OrganizationType'].astype(object)

    le = LabelEncoder()
    enc_dct = dict()
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            idx = df.columns.get_loc(col)
            cat_cols.append(idx)

            df[col] = le.fit_transform(df[col])
            enc_dct[col] = le

    X = df.values
    enc = OneHotEncoder(categorical_features=cat_cols)
    X = enc.fit_transform(X)

    return X

def run_mca(X):
    mca_ben = mca.MCA(X)
    return mca_ben

def run_tsne(X):
    model = TSNE(n_components=6, method='exact', random_state=0)
    X = model.fit_transform(X)
    return X

def run_pca(X):
    pca = PCA(n_components=8, random_state=42)
    X_transformed = pca.fit_transform(X)
    return pca, X_transformed

def prep_kproto(df_):
    df = df_.copy()

    cols_to_keep = ['NumSims',
                    'UserRole',
                    'OrganizationType',
                    'Sample',
                    'ImportedWind',
                    'ImportedSolar',
                    'GenCostMultiLines',
                    'WindCostMultiLines',
                    'BatCostMultiLines',
                    'PvCostMultiLines',
                    'ConCostMultiLines',
                    ]

    df = df[cols_to_keep]

    # convert 1/0 int columns to boolen type
    bool_cols = ['Sample', 'ImportedWind', 'ImportedSolar', 'GenCostMultiLines', 'WindCostMultiLines', 'BatCostMultiLines', 'PvCostMultiLines', 'ConCostMultiLines']
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    # determine which columns are categorical
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == object or df[col].dtype == bool:
            idx = df.columns.get_loc(col)
            cat_cols.append(idx)

    # scale continuous variables
    x_scaled = scale(df['NumSims'].values)
    df['NumSims'] = x_scaled

    return cat_cols, df

def prep_kmodes(df_):
    df = df_.copy()

    cols_to_keep = ['UserRole',
                    'OrganizationType',
                    'Sample',
                    'ImportedWind',
                    'ImportedSolar',
                    'GenCostMultiLines',
                    'WindCostMultiLines',
                    'BatCostMultiLines',
                    'PvCostMultiLines',
                    'ConCostMultiLines',
                    ]

    df = df[cols_to_keep]

    df['UserRole'] = df['UserRole'].astype(object)
    df['OrganizationType'] = df['OrganizationType'].astype(object)

    le = LabelEncoder()
    enc_dct = dict()
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == object:
            idx = df.columns.get_loc(col)
            cat_cols.append(idx)

            df[col] = le.fit_transform(df[col])
            enc_dct[col] = le

    return df

def kproto_cluster(X, cat_cols, init_method='Huang', n_clusters=4):
    kp = kprototypes.KPrototypes(n_clusters=n_clusters, init=init_method, n_init=5, max_iter=5, verbose=2)
    labels = kp.fit_predict(X, categorical=cat_cols)
    return labels

def kmodes_cluster(X, init_method='Huang', n_clusters=4):
    km = kmodes.KModes(n_clusters=n_clusters, n_init=5, init=init_method, verbose=1)
    labels = km.fit_predict(X)
    return labels

def plot_clusters(X, labels, n_clusters=4):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    colors = cm.spectral(labels.astype(float) / n_clusters)
    ax.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)

    plt.savefig('img/pca_clusters.png'.format(n_clusters), dpi=200)
    plt.close()


if __name__ == '__main__':
    plt.close('all')
    # ---Read in data---
    df = pd.read_pickle('data/df.pkl')
    df_users = pd.read_pickle('data/df_users.pkl')

    # CLUSTER AND APPEND LABELS TO USER DATASET
    # ---One Hot Encode---
    X_users = one_hot(df_users)

    # ---Gaussian Mixture---
    # gm = GaussianMixture(n_components=4, covariance_type='tied', max_iter=20, n_init=50, random_state=42, verbose=1)
    # gm.fit(X_users.todense())
    # gm_labels_users = gm.predict(X_users.todense())

    # ---KModes---
    # df_users_km = prep_kmodes(df_users)
    # km_labels_users = kmodes_cluster(df_users_km)

    # ---Agglomerative Clustering---
    # ag = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='average')
    # ag_labels_users = ag.fit_predict(X_users.todense())

    # ---Append lables to dataframe and pickle---
    # df_users['GM_Cluster'] = gm_labels_users
    # df_users['KM_Cluster'] = km_labels_users
    # df_users['AG_Cluster'] = ag_labels_users

    # df_users.to_pickle('data/user_df_clustered.pkl')

    # CLUSTER AND APPEND LABELS TO FULL DATASET
    # ---One Hot Encode---
    X = one_hot(df)

    # ---Gaussian Mixture---
    # gm = GaussianMixture(n_components=4, covariance_type='tied', max_iter=20, n_init=30, random_state=42, verbose=1)
    # gm.fit(X.todense())
    # gm_labels = gm.predict(X.todense())

    # ---KModes---
    # df_km = prep_kmodes(df)
    # km_labels = kmodes_cluster(df_km)

    # ---Agglomerative Clustering--- (Need to run this on AWS)
    # ag = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='average')
    # ag_labels = ag.fit_predict(X.todense())

    # ---Append lables to dataframe and pickle---
    # df['GM_Cluster'] = gm_labels
    # df['KM_Cluster'] = km_labels
    # df['AG_Cluster'] = ag_labels # unable to do AG; not enough RAM

    # df.to_pickle('data/df_clustered.pkl')

    # CREATE SILHOUETTE PLOTS
    # Silhouette plots for user clustered data
    # for i in range(3,9):
    #     for mod in ['AG', 'KM', 'GM']:
    #         cluster_and_plot(X_users.todense(), n_clusters=i, model=mod)

    # See how silhouette score changes with different n_clusters
    for mod in ['AG', 'KM', 'GM']:
        plot_sil_scores(X_users.todense(), model=mod)

    # Silhouette plots for clustered data
    # for i in range(3,9):
    #     for mod in ['AG', 'KM', 'GM']:
    #         cluster_and_plot(X.todense(), n_clusters=i, model=mod, title='full')

    # See how silhouette score changes with different n_clusters
    for mod in ['AG', 'KM', 'GM']:
        plot_sil_scores(X.todense(), model=mod, title='full')
