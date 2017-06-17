from flask import Flask, request, render_template, send_file
import pandas as pd
import folium
app = Flask(__name__)

df = pd.read_pickle('../data/df_clustered.pkl')

C1, C2, C3, C4 = df[df['Cluster'] == 1], df[df['Cluster'] == 2], df[df['Cluster'] == 3], df[df['Cluster'] == 4]

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
    df = pd.read_pickle('../data/df_usa.pkl')
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

    simsdata = pd.DataFrame(df['GEO_ID'].value_counts().astype(float))
    simsdata = simsdata.reset_index()
    simsdata.columns = ['ID', 'Number']

    county_geo = r'../data/us_counties_20m_topo.json'

    m = folium.Map(location=[48, -99], zoom_start=4)
    m.choropleth(geo_path=county_geo,
                    data=simsdata,
                    columns=['ID', 'Number'],
                    key_on='feature.id',
                    fill_color='PuRd',
                    fill_opacity=0.7,
                    line_opacity=0.3,
                    legend_name='Number of Simulations',
                    topojson='objects.us_counties_20m')

    m.save('static/img/choro_map.html')

def marker_cluster_map(df, country, c_num):
    df = df[df['Country'] == country]
    if c_num != 0:
        df = df[df['Cluster'] == c_num]

    centers = pd.read_pickle('../data/centers.pkl')

    center_lat = centers.loc[centers['ISO3136'] == country, 'LAT'].tolist()[0]
    center_lng = centers.loc[centers['ISO3136'] == country, 'LONG'].tolist()[0]

    m = folium.Map(location=[center_lat, center_lng], zoom_start=5.5, control_scale=True)

    # create a marker cluster
    marker_cluster = folium.MarkerCluster('Simulations Cluster').add_to(m)

    latitude = df.Latitude.values
    longitude = df.Longitude.values
    lat_lng = list(zip(latitude, longitude))

    for idx, (lat, lng) in enumerate(lat_lng):
        folium.Marker(location=[lat, lng]).add_to(marker_cluster)

    m.save('static/img/marker_cluster.html')

def usersims_by_cluster(df):
    pt = pd.pivot_table(df, values=['Created'], index=['User'], columns=['Cluster'], aggfunc='count', fill_value=0)

    cluster_df = pd.DataFrame(pt.iloc[:, 0].index)
    cluster_df['Cluster 1'] = pt.iloc[:, 0].values
    cluster_df['Cluster 2'] = pt.iloc[:, 1].values
    cluster_df['Cluster 3'] = pt.iloc[:, 2].values
    cluster_df['Cluster 4'] = pt.iloc[:, 3].values

    output = round(cluster_df.describe().loc[['mean', 'std', 'max']], 2)
    return output.to_html()

# home page
@app.route('/', methods=['GET'])
def index():
    user_table = user_counts(df)
    org_table = org_counts(df)

    df_gen = search_space_counts('MultiGenSearch')
    df_wind = search_space_counts('MultiWindSearch')
    df_bat = search_space_counts('MultiBatSearch')
    df_pv = search_space_counts('MultiPvSearch')
    df_con = search_space_counts('MultiConSearch')

    sims_by_cluster = usersims_by_cluster(df)

    return render_template('index.html', user_table=user_table, org_table=org_table, df_gen=df_gen, df_wind=df_wind, df_bat=df_bat, df_pv=df_pv, df_con=df_con, sims_by_cluster=sims_by_cluster)

# choro map page
@app.route('/show_choro_map', methods=['POST'])
def show_choro_map():
    cluster = int(request.form['input_num_choro'])
    choropleth_map(cluster)

    return send_file('static/img/choro_map.html')

# marker map page
@app.route('/show_marker_map', methods=['POST'])
def show_marker_map():
    c_num = int(request.form['input_num_marker'])
    country = str(request.form['input_country'])
    marker_cluster_map(df, country, c_num)

    return send_file('static/img/marker_cluster.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8105, debug=True, threaded=True)
