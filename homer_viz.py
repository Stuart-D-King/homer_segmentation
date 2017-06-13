import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import folium
import pdb
import seaborn as sns
import scipy.stats as scs
from homer_clean_data import remove_outliers

def bar_charts(df):
    cols = ['ImportedWind', 'ImportedSolar', 'ElectricNotDefault', 'GeneratorNotDefault', 'GenCapCost', 'BatCapCost', 'WindCapCost', 'PvCapCost']
    clusters = np.unique(df['KM_Cluster'])
    fig, axes = plt.subplots(2,4,figsize=(12,7))
    for ax, col in zip(axes.ravel(), cols):
        grp = df.groupby('KM_Cluster')[col].mean()
        ax.bar(clusters, grp, alpha=0.4, color='b')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Average ({})'.format(col))
        ax.set_xticks(range(0, 5))

    # fig.suptitle('Average Number of Simulations Using a Particular Feature')
    fig.tight_layout()
    # fig.subplots_adjust(top=0.9)
    plt.show()

def heat_map_users(df):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    agg = df.groupby(['KM_Cluster', 'UserRole'])['NumSims'].mean()
    ax = sns.heatmap(agg.unstack(level='UserRole'), annot=True)
    ax.set_title('Average Number of Simulations by Cluster and User Role', fontsize=14)

    plt.tight_layout()
    plt.savefig('img/user_heatmap.png', dpi=200)
    # plt.show()

def heat_map_sims(df):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    agg = df.groupby(['KM_Cluster', 'UserRole'])['UserRole'].count()
    ax = sns.heatmap(agg.unstack(level='UserRole'), annot=True)
    ax.set_title('Total Simulations by Cluster and User Role', fontsize=14)

    plt.show()

def count_sims_cluster(df):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    sns.countplot(y='KM_Cluster', data=df, ax=ax, color="c")
    plt.title('Number of Simulations per Cluster', fontsize=14)

    plt.tight_layout()
    plt.savefig('img/cluster_counts.png', dpi=200)
    # plt.show()

def count_user_cluster(df):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    sns.countplot(x='KM_Cluster', hue='UserRole', data=df, palette="Greens_d", ax=ax)

    plt.title('Number of Simulations by Cluster and User Role', fontsize=14)
    plt.show()

def hist_changed_inputs(df):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    ax.hist(df['NumChangedInputs'], bins=20)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Changed Inputs')
    ax.set_title('Histogram of Inputs Changed by User', fontsize=14)
    plt.show()

def hist_sims(df):
    sims = df['NumSims'].values
    mu = sims.mean()
    std = sims.std()
    sims = [s for s in sims if s <= mu + std and s >= mu - std]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    ax.hist(sims, bins=20)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Simulations')
    ax.set_title('Histogram of Simulations by User', fontsize=14)
    plt.show()

def time_series(df):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    date_count = df.groupby(df['Created'].dt.date)['User'].count()
    ax.plot(date_count)
    ax.set_ylabel('Number of Simulations')
    ax.set_title('HOMER Simualtions (October 2104 - April 2017)', fontsize=14)
    plt.show()

def time_hist(df):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)

    date_count = df.groupby(df['Created'].dt.date)['User'].count()
    ax.hist(date_count, color='b', alpha=0.5, edgecolor='k', bins=30, normed=True)
    density = scs.kde.gaussian_kde(date_count.values)
    x_vals = np.linspace(date_count.values.min(), date_count.values.max(), 100)
    kde_vals = density(x_vals)
    ax.plot(x_vals, kde_vals, 'b-')

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Simulations')
    ax.set_title('Histogram of the Number of Simulations', fontsize=14)
    plt.show()

def weekday_weekend(df):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)

    weekday = df[df['Created'].dt.weekday <= 4]
    weekend = df[df['Created'].dt.weekday > 4]
    weekday_count = weekday.groupby(weekday['Created'].dt.date)['User'].count()
    weekend_count = weekend.groupby(weekend['Created'].dt.date)['User'].count()

    ax.hist(weekday_count.values, color='b', alpha=0.5, edgecolor='k', bins=30, normed=True)

    density = scs.kde.gaussian_kde(weekday_count.values)
    x_vals = np.linspace(weekday_count.values.min(), weekday_count.values.max(), 100)
    kde_vals = density(x_vals)
    ax.plot(x_vals, kde_vals, 'b-', label='weekday')

    ax.hist(weekend_count.values, color='g', alpha=0.5, edgecolor='k', bins=30, normed=True)

    density = scs.kde.gaussian_kde(weekend_count.values)
    x_vals = np.linspace(weekend_count.values.min(), weekend_count.values.max(), 100)
    kde_vals = density(x_vals)
    ax.plot(x_vals, kde_vals, 'g-', label='weekend')

    ax.set_xlabel('Simulations')
    ax.set_ylabel('Frequency (Normed)')
    ax.set_title('Histogram of Weekend and Weekday Simulations', fontsize=14)

    plt.legend()
    plt.show()

def cluster_bars(df):
    # categorical vs categorical vs numeric
    agg = df.groupby(['KM_Cluster', 'UserRole'])['UserRole'].count()
    # print(agg)
    agg = agg.unstack(level='UserRole')
    # print(agg)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    agg.plot(kind='bar', ax=ax).set_ylabel('Simulations')

    plt.title('Number of Simulations by Cluster and User Role', fontsize=14)

    plt.savefig('img/sims_by_cluser_user.png', dpi=200)
    plt.close()

def marker_map(df, title=None):
    df = df[df['KM_Cluster'] == 2]

    sizes = df.NumSims.values
    m = folium.Map(location=[51.513, -0.137], zoom_start=3, tiles='Cartodb Positron')

    latitude = df.Latitude.values
    longitude = df.Longitude.values
    lat_lng = list(zip(latitude, longitude))
    for idx, (lat, lng) in enumerate(lat_lng):
        folium.CircleMarker(location=[lat, lng], color='rgba(44, 185, 34, 1)', fill_color='rgba(44, 185, 34, 1)', radius=sizes[idx]).add_to(m)

    if title is not None:
        m.save('{}.html'.format(title))

    marker_map.save('img/maps/marker_map.html')


def choropleth_map(df, title=None):

    def set_id_(fips):
        '''Modify FIPS code to match GeoJSON property'''
        if fips == '0':
            return None
        elif len(fips) <= 4:
            return ''.join(['0500000US0', fips])
        else:
            return ''.join(['0500000US', fips])

    df['FIPS'] = df['FIPS'].astype(str)
    df['GEO_ID'] = df['FIPS'].apply(set_id_)
    df = df.dropna()

    state_geo = r'data/us-states.json'
    county_geo = r'data/us_counties_20m_topo.json'
    # data = df['NumSims'].values
    # 'data/US_Unemployment_Oct2012.csv'

    m = folium.Map(location=[48, -99], zoom_start=4)
    m.choropleth(geo_path=county_geo,
                    data=df,
                    columns=['GEO_ID', 'NumSims'],
                    key_on='feature.id',
                    fill_color='PuRd',
                    fill_opacity=0.7,
                    line_opacity=0.3,
                    legend_name='Number of Simulations',
                    topojson='objects.us_counties_20m')

    m.save('img/maps/choro_map.html')

def marker_cluster_map(df):
    # df = df[df['Cluster'] == 2]

    m = folium.Map(location=[51.513, -0.137], zoom_start=3, control_scale=True)

    # create a marker cluster
    marker_cluster = folium.MarkerCluster('Simulations Cluster').add_to(m)

    latitude = df.Latitude.values
    longitude = df.Longitude.values
    lat_lng = list(zip(latitude, longitude))
    colors = [cm.spectral(float(i) / 5) for i in range(5)]

    for idx, (lat, lng) in enumerate(lat_lng):
        folium.Marker(location=[lat, lng], icon=folium.Icon(color=colors[df['Cluster'][idx]-1])).add_to(marker_cluster)

    m.save('img/maps/marker_cluster.html')


if __name__ == '__main__':
    plt.close('all')
    df_users = pd.read_pickle('data/df_users_clustered.pkl')
    df = pd.read_pickle('data/df_clustered.pkl')
    # df_users_usa = pd.read_pickle('data/df_users_usa.pkl')
    # df_users_usa = pd.read_pickle('data/df_users_usa.pkl')
    # df_usa = pd.read_pickle('data/df_usa.pkl')

    # create_map(df_users, title='first_map')
    # choropleth_map(df_usa)
    marker_cluster_map(df)
    # count_sims_cluster(df)
    # count_user_cluster(df)
    # bar_charts(df)
    # heat_map_users(df_users)
    # heat_map_s ims(df)
    # hist_sims(df_users)
    # hist_changed_inputs(df_users)
    # time_series(df)
    # time_hist(df)
    # weekday_weekend(df)

    # c0 = df[df['KM_Cluster'] == 0]
    # c1 = df[df['KM_Cluster'] == 1]
    # c2 = df[df['KM_Cluster'] == 2]
    # c3 = df[df['KM_Cluster'] == 3]
    # c4 = df[df['KM_Cluster'] == 4]
    # time_hist(c4)

    # df_cleaned = remove_outliers(df_users_usa)
    # choropleth_map(df_cleaned)
    # cluster_bars(df_users)
