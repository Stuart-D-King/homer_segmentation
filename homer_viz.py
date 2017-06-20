import pdb
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scs
import folium
import plotly
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import seaborn as sns


def bar_charts(df):
    '''
    Create bar chart graph of ImportedWind, ImportedSolar, and Sample variables.

    :param df: dataframe from which data is pulled
    '''
    cols = ['ImportedWind', 'ImportedSolar', 'Sample']
    clusters = np.unique(df['Cluster'])
    fig, axes = plt.subplots(1,3,figsize=(10,3.5))

    for ax, col in zip(axes.ravel(), cols):
        grp = df.groupby('Cluster')[col].mean()
        ax.bar(clusters, grp, alpha=0.4, color='b')
        ax.set_xlabel('')
        ax.set_ylabel('Average ({})'.format(col), fontsize=16)
        ax.set_xticks(range(1, 5))
        ax.tick_params(labelsize=14)
        ax.axhline(y=np.mean(df[col]), color='red', linestyle='--')

    fig.tight_layout()
    plt.savefig('img/imports.png')
    plt.close()


def heat_map_users(df):
    '''
    Create heat map of the number of simulations by user role and cluster.

    :param df: dataframe from which data is pulled
    '''
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    agg = df.groupby(['Cluster', 'UserRole'])['NumSims'].mean()
    ax = sns.heatmap(agg.unstack(level='UserRole'), annot=True)
    ax.set_title('Average Number of Simulations by Cluster and User Role', fontsize=14)

    plt.tight_layout()
    plt.yticks(rotation='horizontal')
    plt.savefig('img/user_heatmap.png', dpi=200)
    plt.close()


def heat_map_sims(df):
    '''
    Create heat map of the total number of simulations by cluster using DefaultGenerator variable.

    :param df: dataframe from which data is pulled
    '''
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    agg = df.groupby(['Cluster', 'DefaultGenerator'])['DefaultGenerator'].count()
    ax = sns.heatmap(agg.unstack(level='DefaultGenerator'), annot=True)
    ax.set_title('Total Simulations by Cluster and DefaultGenerator', fontsize=14)

    plt.tight_layout()
    plt.yticks(rotation='horizontal')
    plt.show()


def count_sims_cluster(df):
    '''
    Create horizontal bar chart of the number of simulations by cluster.

    :param df: dataframe from which data is pulled
    '''
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    sns.countplot(y='Cluster', data=df, ax=ax, color="c")

    ax.set_xlabel('Count', fontsize=14)
    ax.set_ylabel('Cluster', fontsize=14)
    ax.tick_params(labelsize=14)
    plt.title('Number of Simulations per Cluster', fontsize=20)

    plt.tight_layout()
    plt.savefig('img/cluster_counts.png', dpi=200)
    plt.close()


def count_user_cluster(df):
    '''
    Create stacked horizontal bar chart of the number of simulations by cluster and UserRole.

    :param df: dataframe from which data is pulled
    '''
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    sns.countplot(x='Cluster', hue='UserRole', data=df, palette="Greens_d", ax=ax)

    plt.title('Number of Simulations by Cluster and User Role', fontsize=20)
    plt.show()


def hist_sims(df):
    '''
    Create histogram of the number of simulations with outliers (+/- 3 standard deviations) removed.

    :param df: dataframe from which data is pulled
    '''
    sims = df['NumSims'].values
    mu = sims.mean()
    std = sims.std()
    sims = [s for s in sims if s <= mu + (std*3) and s >= mu - (std*3)]
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    ax.hist(sims, bins=50)
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Simulations')
    ax.set_title('Histogram of Simulations by User', fontsize=14)
    plt.savefig('img/hist_sims.png', dpi=200)
    plt.close()


def time_series(df):
    '''
    Create time series line graph of the number of simulations each day from April 2014 to April 2017.

    :param df: dataframe from which data is pulled
    '''
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    date_count = df.groupby(df['Created'].dt.date)['User'].count()

    ax.plot(date_count)
    ax.set_ylabel('Number of Simulations')
    ax.set_title('HOMER Simualtions (April 2014 - April 2017)', fontsize=14)

    plt.show()


def time_hist(df):
    '''
    Create histogram of the number of simulations per day from April 2014 to April 2017.

    :param df: dataframe from which data is pulled
    '''
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


def weekday_weekend(df_):
    '''
    Create a weekday and weekend histogram for each cluster.

    :param df: dataframe from which data is pulled
    '''
    clusters = [1,2,3,4]
    fig, axes = plt.subplots(2,2,figsize=(8,8))
    for ax, c in zip(axes.ravel(), clusters):
        df = df_[df_['Cluster'] == c]
        weekday = df[df['Created'].dt.weekday <= 4]
        weekend = df[df['Created'].dt.weekday > 4]
        weekday_count = weekday.groupby(weekday['Created'].dt.date)['User'].count()
        weekend_count = weekend.groupby(weekend['Created'].dt.date)['User'].count()

        ax.hist(weekday_count.values, color='b', alpha=0.5, bins=30, label='weekday')#, normed=True)

        # density = scs.kde.gaussian_kde(weekday_count.values)
        # x_vals = np.linspace(weekday_count.values.min(), weekday_count.values.max(), 100)
        # kde_vals = density(x_vals)
        # ax.plot(x_vals, kde_vals, 'b-', label='weekday')

        ax.hist(weekend_count.values, color='g', alpha=0.5, bins=30, label='weekend')#, normed=True)

        # density = scs.kde.gaussian_kde(weekend_count.values)
        # x_vals = np.linspace(weekend_count.values.min(), weekend_count.values.max(), 100)
        # kde_vals = density(x_vals)
        # ax.plot(x_vals, kde_vals, 'g-', label='weekend')

        ax.set_xlabel('Simulations')
        ax.set_ylabel('Frequency')
        ax.set_title('Cluster {}'.format(c), fontsize=12)
        ax.legend()

    plt.tight_layout()
    plt.savefig('img/weekday_weekend.png', dpi=200)


def cluster_bars_org(df):
    '''
    Create bar chart of the number of simulations by organization type in each cluster.

    :param df: dataframe from which data is pulled
    '''
    # categorical vs categorical vs numeric
    agg = df.groupby(['Cluster', 'OrganizationType'])['OrganizationType'].count()
    agg = agg.unstack(level='OrganizationType')
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    agg.plot(kind='bar', ax=ax).set_ylabel('Simulations')
    plt.title('Number of Simulations by Cluster and Organization Type', fontsize=16)
    plt.xticks(rotation='horizontal')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
    # ax.tick_params(labelsize=14)
    ax.set_xlabel('')

    plt.savefig('img/sims_by_cluster_org.png', dpi=200)
    plt.close()


def cluster_bars_user(df):
    '''
    Create bar chart of the number of simulations by user role in each cluster.

    :param df: dataframe from which data is pulled
    '''
    # categorical vs categorical vs numeric
    agg = df.groupby(['Cluster', 'UserRole'])['UserRole'].count()
    agg = agg.unstack(level='UserRole')
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    agg.plot(kind='bar', ax=ax).set_ylabel('Simulations', fontsize=20)
    plt.title('Number of Simulations by Cluster and User Role', fontsize=24)
    ax.tick_params(labelsize=20)
    plt.xticks(rotation='horizontal')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize=16, fancybox=True, shadow=True, ncol=4)
    # ax.tick_params(labelsize=14)
    ax.set_xlabel('')
    plt.legend(loc='best', fontsize=18)
    plt.tight_layout()

    plt.savefig('img/sims_by_cluster_user.png', dpi=600)
    plt.close()


def marker_map(df, c_num=0):
    '''
    Create a marker map of for where each simulation is run.

    :param df: dataframe from which data is pulled
    :param c_num: the cluster to map; default = 0, which means all clusters are mapped
    '''
    if c_num != 0:
        df = df[df['Cluster'] == cluster]

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


def choropleth_map(df, c_num=0):
    '''
    Create a choropleth (heat map) of simulations in the United States by county.

    :param df: dataframe from which data is pulled (should be a dataframe of U.S. simulations)
    :param c_num: the cluster to map; default = 0, which means all clusters are mapped
    '''
    if c_num != 0:
        df = df[df['Cluster'] == cluster]

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
    simsdata = pd.DataFrame(df['GEO_ID'].value_counts().astype(float))
    simsdata = simsdata.reset_index()
    simsdata.columns = ['ID', 'Number']
    simsrange = np.max(simsdata['Number']) - 0
    threshold_scale = [0, simsrange*.2, simsrange*.4, simsrange*.6, simsrange*.8, np.max(simsdata['Number'])]
    county_geo = r'data/us_counties_20m_topo.json'
    m = folium.Map(location=[48, -99], zoom_start=4)
    m.choropleth(geo_path=county_geo,
                    data=simsdata,
                    columns=['ID', 'Number'],
                    key_on='feature.id',
                    threshold_scale=threshold_scale,
                    fill_color='PuRd',
                    fill_opacity=0.7,
                    line_opacity=0.3,
                    legend_name='Number of Simulations',
                    topojson='objects.us_counties_20m')

    m.save('img/maps/choro_map.html')


def marker_cluster_map(df_, country, cluster=0):
    '''
    Create a marker cluster map of simulations in a particular country.

    :param df: dataframe from which data is pulled
    :param country: country for which to map simulations
    :param cluster: the cluster to map; default = 0, which means all clusters are mapped
    '''
    centers = pd.read_pickle('data/centers.pkl')
    df = df_[df_['Country'] == country]

    if cluster != 0:
        df = df_[df_['Cluster'] == cluster]

    center_lat = centers.loc[centers['ISO3136'] == country, 'LAT'].tolist()[0]
    center_lng = centers.loc[centers['ISO3136'] == country, 'LONG'].tolist()[0]
    m = folium.Map(location=[center_lat, center_lng], zoom_start=6, max_zoom=10, control_scale=True)
    # create a marker cluster
    marker_cluster = folium.MarkerCluster('Simulations Cluster').add_to(m)
    latitude = df.Latitude.values
    longitude = df.Longitude.values
    lat_lng = list(zip(latitude, longitude))

    for idx, (lat, lng) in enumerate(lat_lng):
        folium.Marker(location=[lat, lng]).add_to(marker_cluster)

    m.save('img/maps/marker_cluster.html')


if __name__ == '__main__':
    plt.close('all')
    df = pd.read_pickle('data/df_clustered.pkl')
    df_usa = pd.read_pickle('data/df_usa.pkl')
    # df_users = pd.read_pickle('data/df_users.pkl')

    # bar_charts(df)
    # count_sims_cluster(df)
    # count_user_cluster(df)
    # heat_map_users(df_users)
    # heat_map_sims(df)
    # hist_sims(df_users)
    # time_series(df)
    # time_hist(df)
    # weekday_weekend(df)
    # cluster_bars_user(df)
    # cluster_bars_org(df)

    # MAPS!!
    # choropleth_map(df_usa)
    # marker_cluster_map(df, 'DE', 3)
