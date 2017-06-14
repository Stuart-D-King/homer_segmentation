from flask import Flask, request, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import plotly
from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import folium
import seaborn as sns
app = Flask(__name__)

df = pd.read_pickle('../data/df.pkl')
df_clustered = pd.read_pickle('../data/df_clustered.pkl')
df_users = pd.read_pickle('../data/df_users.pkl')
df_users_clustered = pd.read_pickle('../data/df_users_clustered.pkl')
C1, C2, C3, C4 = df_clustered[df_clustered['Cluster'] == 1], df_clustered[df_clustered['Cluster'] == 2], df_clustered[df_clustered['Cluster'] == 3], df_clustered[df_clustered['Cluster'] == 4]

def user_counts(df):
    grps = df.UserRole.value_counts(dropna=False)
    pct = [x / float(sum(grps)) for x in grps]
    s1 = pd.Series(grps.values, index=grps.index, name='Count')
    s2 = pd.Series(pct, index=grps.index, name='Total%')
    s2 = s2.apply(lambda x: round(x*100, 2))
    df_grp = pd.concat([s1, s2], axis=1)
    return df_grp.to_html()

def org_counts(df):
    grps = df.OrganizationType.value_counts(dropna=False)
    pct = [x / float(sum(grps)) for x in grps]

    s1 = pd.Series(grps.values, index=grps.index, name='Count')
    s2 = pd.Series(pct, index=grps.index, name='Total%')
    s2 = s2.apply(lambda x: round(x*100, 2))
    df_grp = pd.concat([s1, s2], axis=1)
    return df_grp.to_html()

def search_space_counts(col):
    frames = []
    clusters = [C1, C2, C3, C4]
    for idx, cluster in enumerate(clusters):
        grps = cluster[col].value_counts(dropna=False)
        pct = [x / float(sum(grps)) for x in grps]

        s1 = pd.Series(grps.values, index=grps.index, name='Count')
        s2 = pd.Series(pct, index=grps.index, name='Total%')
        s2 = s2.apply(lambda x: round(x*100, 2))

        df_grp = pd.concat([s1, s2], axis=1)
        df_grp =df_grp.reindex(['True', 'False', 'NA'])
        frames.append(df_grp)

    df_combo = pd.concat(frames, axis=1)
    vals = df_combo.values

    l1 = ['Cluster 1', 'Cluster 1', 'Cluster 2', 'Cluster 2', 'Cluster 3', 'Cluster 3', 'Cluster 4', 'Cluster 4']
    l2 = ['Count', 'Total%', 'Count', 'Total%', 'Count', 'Total%', 'Count', 'Total%']

    tuples = list(zip(l1, l2))
    index = pd.MultiIndex.from_tuples(tuples, names=[col, ''])

    df = pd.DataFrame(vals, index=['True', 'False', 'NA'], columns=index)
    return df.to_html()

def choropleth_map(cluster=0):
    df = pd.read_pickle('../data/df_users_usa.pkl')
    if cluster != 0:
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

    county_geo = r'../data/us_counties_20m_topo.json'

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

    m.save('static/img/choro_map.html')

# home page
@app.route('/', methods=['GET'])
def index():
    users_table = user_counts(df)
    org_table = org_counts(df)

    df_gen = search_space_counts('MultiGenSearch')
    df_wind = search_space_counts('MultiWindSearch')
    df_bat = search_space_counts('MultiBatSearch')
    df_pv = search_space_counts('MultiPvSearch')
    df_con = search_space_counts('MultiConSearch')

    return render_template('index.html', user_table=user_table, org_table=org_table, df_gen=df_gen, df_wind=df_wind, df_bat=df_bat, df_pv=df_pv, df_con=df_con)

# map page
@app.route('/show_map', methods=['POST'])
def show_map():
    cluster = int(request.form['user_input'])
    choropleth_map(cluster)
    return send_file('static/img/choro_map.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True, threaded=True)
