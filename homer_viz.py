import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import folium
import pdb


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
    df = df[df['KM_Cluster'] == 2]

    m = folium.Map(location=[51.513, -0.137], zoom_start=3, control_scale=True)

    # create a marker cluster
    marker_cluster = folium.MarkerCluster('Simulations Cluster').add_to(m)

    latitude = df.Latitude.values
    longitude = df.Longitude.values
    lat_lng = list(zip(latitude, longitude))

    for idx, (lat, lng) in enumerate(lat_lng):
        folium.Marker(location=[lat, lng]).add_to(marker_cluster)

    m.save('img/maps/marker_cluster.html')


if __name__ == '__main__':
    df_users = pd.read_pickle('data/df_users_clustered.pkl')
    df = pd.read_pickle('data/df_clustered.pkl')
    df_users_usa = pd.read_pickle('data/df_users_usa.pkl')
    df_usa = pd.read_pickle('data/df_usa.pkl')

    # create_map(df_users, title='first_map')
    # choropleth_map(df_usa)
    # marker_cluster_map(df_users)
