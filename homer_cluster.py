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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances


def one_hot(df_):
    df = df_.copy()

    cols_to_keep = ['UserRole',
                    'OrganizationType',
                    'MultiGenSearch',
                    'MultiWindSearch',
                    'MultiBatSearch',
                    'MultiPvSearch',
                    'MultiConSearch',
                    'Sample',
                    'DefaultGenerator',
                    'ImportedWind',
                    'ImportedSolar'
                    ]

    df = df[cols_to_keep]

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
    X = mca_ben.fs_r(1)
    return mca_ben, X

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
                    'MultiGenSearch',
                    'MultiWindSearch',
                    'MultiBatSearch',
                    'MultiPvSearch',
                    'MultiConSearch',
                    'Sample',
                    'DefaultGenerator',
                    'ImportedWind',
                    'ImportedSolar'
                    ]

    df = df[cols_to_keep]

    # convert 1/0 int columns to boolen type
    bool_cols = ['ImportedWind', 'ImportedSolar', 'Sample']

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
                    'MultiGenSearch',
                    'MultiWindSearch',
                    'MultiBatSearch',
                    'MultiPvSearch',
                    'MultiConSearch',
                    'Sample',
                    'DefaultGenerator',
                    'ImportedWind',
                    'ImportedSolar'
                    ]

    df = df[cols_to_keep]

    # df['UserRole'] = df['UserRole'].astype(object)
    # df['OrganizationType'] = df['OrganizationType'].astype(object)

    le = LabelEncoder()
    enc_dct = dict()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = le.fit_transform(df[col])
            enc_dct[col] = le

    return df

def run_kproto(X, cat_cols, init_method='Cao', n_clusters=4):
    kp = kprototypes.KPrototypes(n_clusters=n_clusters, init=init_method, n_init=10, max_iter=5, verbose=2)
    labels = kp.fit_predict(X, categorical=cat_cols)
    return kp, labels

def run_kmodes(X, init_method='Huang', n_clusters=4):
    km = kmodes.KModes(n_clusters=n_clusters, n_init=10, init=init_method, verbose=1)
    labels = km.fit_predict(X)
    return km, labels

def plot_clusters(X, labels, n_clusters=4):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    colors = cm.spectral(labels.astype(float) / n_clusters)
    ax.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors)

    plt.savefig('img/plotted_clusters.png'.format(n_clusters), dpi=200)
    plt.close()


if __name__ == '__main__':
    plt.close('all')
    # ---Read in data---
    df = pd.read_pickle('data/df.pkl')
    df_users = pd.read_pickle('data/df_users.pkl')

    # ---KModes---
    df_users_km = prep_kmodes(df_users)
    km_model, km_labels = run_kmodes(df_users_km)

    # ---Append lables to dataframe and pickle---
    df_users['Cluster'] = km_labels
    df_users['Cluster'] = df_users['Cluster'].apply(lambda x: x+1)

    # ---Map cluster labels to full dataframe---
    keys = df_users['UserId'].tolist()
    values = df_users['Cluster'].tolist()
    user_cluster_dct = dict(zip(keys, values))
    df['Cluster'] = df['User'].map(user_cluster_dct)

    # ---Pickle clustered dataframes and KModes model---
    df_users.to_pickle('data/df_users_clustered.pkl')
    df.to_pickle('data/df_clustered.pkl')

    with open('data/km_model.pkl', 'wb') as f:
        pickle.dump(km_model, f)

    # ---Finally, read back in dataframes with cluster labels---
    df_clustered = pd.read_pickle('data/df_clustered.pkl')
    df_users_clustered = pd.read_pickle('data/df_users_clustered.pkl')


    # EXTRA STUFF
    # ---One Hot Encode---
    # X_users = one_hot(df_users)

    # ---Gaussian Mixture---
    # gm = GaussianMixture(n_components=4, covariance_type='tied', max_iter=20, n_init=50, random_state=42, verbose=1)
    # gm.fit(X_users.todense())
    # gm_labels = gm.predict(X_users.todense())

    # ---Agglomerative Clustering---
    # ag = AgglomerativeClustering(n_clusters=4, affinity='cosine', linkage='average')
    # ag_labels = ag.fit_predict(X_users.todense())

    # ---Append lables to dataframe and pickle---
    # df_users['GM_Cluster'] = gm_labels
    # df_users['AG_Cluster'] = ag_labels
