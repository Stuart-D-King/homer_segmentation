import numpy as np
import pandas as pd
from kmodes import kmodes, kprototypes
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mca
import pickle
import pdb
from homer_silhouette import cluster_and_plot


def prep_kproto(main_df):
    df = main_df.copy()

    cols_to_keep = ['NumSims',
                    'StyleRealtime',
                    'StyleSimple',
                    'StyleScheduled',
                    'MonthlyPurchaseCapacity',
                    'UnreliableGrid',
                    'Sample',
                    'Latitude',
                    'Longitude',
                    'ImportedWind',
                    'ImportedSolar',
                    'GenCostMultiLines',
                    'WindCostMultiLines',
                    'BatCostMultiLines',
                    'PvCostMultiLines',
                    'ConCostMultiLines',
                    'IsProUser',
                    'DaysSinceFirst',
                    'AcademicOrIndividual'
                    ]

    df = df[cols_to_keep]

    # convert 1/0 int columns to boolen type
    bool_cols = ['StyleRealtime', 'StyleSimple', 'StyleScheduled', 'MonthlyPurchaseCapacity', 'UnreliableGrid', 'Sample', 'Chp', 'GridConnected','ImportedWind', 'ImportedSolar', 'GenCostMultiLines', 'WindCostMultiLines', 'BatCostMultiLines', 'PvCostMultiLines', 'ConCostMultiLines', 'IsProUser', 'AcademicOrIndividual']
    for col in bool_cols:
        df[col] = df[col].astype(bool)

    # determine which columns are categorical
    cat_cols = []
    for col in bool_cols:
        idx = df.columns.get_loc(col)
        cat_cols.append(idx)

    # scale continuous variables
    for col in cols_to_keep:
        if col not in bool_cols:
            x_scaled = scale(df[col].values)
            df[col] = x_scaled

    kproto_data = df.values

    return cat_cols, kproto_data

def prep_kmodes(main_df):
    df = main_df.copy()

    cols_to_keep =['StyleExtension',
                    'StyleRealtime',
                    'StyleSimple',
                    'StyleScheduled',
                    'MonthlyPurchaseCapacity',
                    'UnreliableGrid',
                    'Sample',
                    'Chp',
                    'GridConnected',
                    'ImportedWind',
                    'ImportedSolar',
                    'GenCostMultiLines',
                    'WindCostMultiLines',
                    'BatCostMultiLines',
                    'PvCostMultiLines',
                    'ConCostMultiLines',
                    'IsProUser',
                    'AcademicOrIndividual'
                    ]

    df = df[cols_to_keep]

    # binary data used for silhouette scoring
    binary_data = df.values

    for col in df.columns.tolist():
        df[col] = df[col].astype(bool)

    # data with categorical values for kmodes algorithm
    kmodes_data = df.values

    return binary_data, kmodes_data

def kproto_cluster(X, cat_cols, init_method='Huang', n_clusters=4):
    kp = kprototypes.KPrototypes(n_clusters=n_clusters, init=init_method, n_init=5, max_iter=5, verbose=2)
    labels = kp.fit_predict(X, categorical=cat_cols)
    return labels

def kmodes_cluster(X, init_method='Huang', n_clusters=4):
    km = kmodes.KModes(n_clusters=n_clusters, n_init=5, init=init_method, verbose=1)
    labels = km.fit_predict(X)
    return labels

# PCA really isn't viable with kmodes or kprototypes
def set_pca(X):
    pca = PCA(n_components=10, random_state=42)
    X_transformed = pca.fit_transform(X)
    return pca, X_transformed

# PCA really isn't viable with kmodes or kprototypes
def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
        color=[(0.949, 0.718, 0.004),
               (0.898, 0.49, 0.016),
               (0.863, 0, 0.188),
               (0.694, 0, 0.345),
               (0.486, 0.216, 0.541),
               (0.204, 0.396, 0.667),
               (0.035, 0.635, 0.459),
               (0.486, 0.722, 0.329),
               ])

    for i in range(num_components):
        ax.annotate(r'%s%%' % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va='bottom', ha='center', fontsize=12)

    ax.set_xticklabels(ind, fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    plt.savefig('img/pca_scree.png', dpi=250)
    plt.close()

# PCA plots aren't adviseable with kprototypes
def plot_kproto(X, labels, n_clusters=4, title=None):
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    colors = cm.spectral(labels.astype(float) / n_clusters)
    ax.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)

    if title is not None:
        plt.title(title, fontsize=16)

    plt.savefig('img/kproto_clusters_{}.png'.format(n_clusters), dpi=250)
    plt.close()

# PCA plots aren't adviseable with kmodes...
def plot_kmodes(X, labels, n_clusters=4, title=None):
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(111)

    colors = cm.spectral(labels.astype(float) / n_clusters)
    ax.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)

    if title is not None:
        plt.title(title, fontsize=16)

    plt.savefig('img/kmodes_clusters_{}.png'.format(n_clusters), dpi=250)
    plt.close()

if __name__ == '__main__':
    plt.close('all')
    # ---Read in data---
    df = pd.read_pickle('data/full_df.pkl')
    df_users = pd.read_pickle('data/user_df.pkl')
    # X_ohe = np.loadtxt('data/users_ohe.csv')

    # ---Run kprototypes clustering---
    # cat_cols, kproto_data = prep_kproto_pca(df_users)
    # kp_labels = kproto_cluster(kproto_data, cat_cols)
    # pca, X_transformed = set_pca(pca_data)
    # plot_kproto(X_transformed, kp_labels)
    # scree_plot(pca)

    # ---Run kmodes clustering---
    # binary_data, kmodes_data = prep_kmodes(df_users)
    # km_labels = kmodes_cluster(kmodes_data, n_clusters=4)
    # pca, X_transformed = set_pca(pca_data.todense())
    # plot_kmodes(X_transformed, km_cluster_labels)
    # scree_plot(pca)

    # ---Create silhoutte plots---
    # for i in range(2,9):
    #     cluster_and_plot(binary_data, n_clusters=i)
