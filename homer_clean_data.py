import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pdb
import itertools
import requests
import reverse_geocoder as rg
import xml.etree.ElementTree as ET

def read_data():
    df = pd.read_csv('data/combined_grid_and_run_data.csv', encoding='iso-8859-1')

    cols = ['UserRole',
            'OrganizationType',
            'Sample',
            'Latitude',
            'Longitude',
            'User',
            'Created',
            'Generator0CostTable',
            'WindTurbine0CostTable',
            'Battery0CostTable',
            'Pv0CostTable',
            'ConverterCostTable',
            'ImportedWind',
            'ImportedSolar']

    df = df[cols]

    print('Cleaning columns...')
    # change type of 'Created' to datetime
    df['Created'] = pd.to_datetime(df['Created'])

    # clean 'UserRole'
    df['UserRole'] = df['UserRole'].replace(['Student', 'Undergraduate Student', 'Post-graduate Student', 'Tenured or Tenure-track Faculty', 'Faculty', 'Research Staff'], 'Academic')
    df['UserRole'] = df['UserRole'].replace(['Engineer', 'Mechanic/Technician/Facility Manager'], 'Technical')
    df['UserRole'] = df['UserRole'].replace(['IT Professional', 'IT Staff', 'Sales/Marketing', 'Purchasing Agent', 'Executive', 'Planner/Regulator/Policy Maker'], 'Business')
    df['UserRole'] = df['UserRole'].replace([np.nan, 'Personal Interest', 'Staff', 'Other'], 'NA')

    df['UserRole'] = df['UserRole'].astype('category')

    # clean 'OrganizaitonType'
    df['OrganizationType'] = df['OrganizationType'].replace([np.nan, 'Other', 'Interested Individual', 'Microgrid End User (all types)'], 'NA')
    df['OrganizationType'] = df['OrganizationType'].replace(['Engineering Services Company', 'Electric Distribution Utility', 'Independent Power Producer', 'Project Developer'], 'Engineering')
    df['OrganizationType'] = df['OrganizationType'].replace(['EPC/Construction Company', 'Solar Installation Company', 'Equipment Vendor'], 'Vendor')
    df['OrganizationType'] = df['OrganizationType'].replace(['Government', 'Non-Governmental Organization (NGO)'], 'Public')
    df['OrganizationType'] = df['OrganizationType'].replace(['Academic Institution or Research Center'], 'Academic')
    df['OrganizationType'] = df['OrganizationType'].replace(['Other Professional Services Company', 'Finance Organization'], 'Service')

    df['OrganizationType'] = df['OrganizationType'].astype('category')

    # clean 'Sample'; if a sample is used = True, else = False
    df['Sample'] = np.where(df['Sample'].notnull(), True, False)

    # clean latitude and longitude columns
    # will look into 'geopy' to see if I can impute coordinates for simulaitons without latitude and longitude
    lat_lon_cols = ['Latitude', 'Longitude']
    for col in lat_lon_cols:
        df[col] = df[col].replace(['0'], np.nan)
        df[col] = df[col].astype(float)

    # clean 'User'; drop users with len() != 6
    df['User'] = df['User'].apply(pd.to_numeric, errors='coerce')
    # drop rows where User is null
    df = df[pd.notnull(df['User'])]
    # convert to int and then str types
    df['User'] = df['User'].astype(int)
    df['User'] = df['User'].astype(str)
    # drop any Users with IDs whose length is not 6
    df = df[df['User'].map(len) == 6]

    print('Creating new boolean columns...')
    # create boolean columns of if a cost table has multiple lines
    cost_tables = ['Generator0CostTable', 'WindTurbine0CostTable', 'Battery0CostTable', 'Pv0CostTable', 'ConverterCostTable']
    costs = ['Gen', 'Wind', 'Bat', 'Pv', 'Con']
    for name, table in zip(costs, cost_tables):
        df[name+'CostMultiLines'] = df[table].astype(str).apply(lambda x: x.split('|')).apply(lambda x: len(x) > 1)

    # drop cost table columns no longer needed
    for table in cost_tables:
        df.drop(table, axis=1, inplace=True)

    print('Converting booleans to integers...')
    # convert boolean columns to int type
    bool_cols = ['Sample', 'ImportedWind', 'ImportedSolar', 'GenCostMultiLines', 'WindCostMultiLines', 'BatCostMultiLines', 'PvCostMultiLines', 'ConCostMultiLines']

    for col in bool_cols:
        if df[col].dtype != bool:
            df[col] = df[col].astype(bool)
        df[col] = df[col].astype(int)

    df.dropna(axis=0, how='any', inplace=True)
    # reset dataframe index
    df.reset_index(drop=True, inplace=True)

    print('All done!')
    return df

def score_rows(df):
    # best possible score = 11
    # worst possible score = -3
    all_scores = []
    for idx, row in df.iterrows():
        score = 0
        if row['UserRole'] == 'Technical':
            score += 2
        elif row['UserRole'] == 'Business':
            score += 1
        elif row['UserRole'] == 'Academic':
            score -= 1

        if row['OrganizationType'] == 'Engineering':
            score += 2
        elif row['OrganizationType'] == 'Public' or row['OrganizationType'] == 'Vendor':
            score += 1
        elif row['OrganizationType'] == 'Academic':
            score -= 1

        if row['Sample'] == 1:
            score += 1

        if row['ImportedWind'] == 1:
            score += 1

        if row['ImportedSolar'] == 1:
            score += 1

        if row['GenCostMultiLines'] == 1:
            score += 1

        if row['WindCostMultiLines'] == 1:
            score += 1

        if row['BatCostMultiLines'] == 1:
            score += 1

        if row['PvCostMultiLines'] == 1:
            score += 1

        if row['ConCostMultiLines'] == 1:
            score += 1

        if row['Latitude'] == np.nan or row['Longitude'] == np.nan:
            score -= 1

        all_scores.append(score)

    return all_scores

def create_user_df(df):
    '''
    Create user dataframe from passed in dataframe, grouping by user. Takes the most frequent occuring value from users with multiple simulations.

    :param df: dataframe to build new dataframe
    :returns: created user dataframe
    '''
    df.dropna(axis=0, how='any', inplace=True)

    # group by user
    users = df.groupby('User')
    # get number of simulations by user
    sims = users['UserRole'].count()
    user_id = users.grouper.result_index.values
    num_sims = sims.values

    # create dataframe of user ids and add number of simulations column
    df_users = pd.DataFrame(data=user_id, columns=['UserId'])
    df_users['NumSims'] = num_sims

    # using the mode for each column from the full data frame, add column values to each new column created in user dataframe
    cols = ['UserRole',
            'OrganizationType',
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
            ]

    for col in cols:
        df_users[col] = users[col].agg(lambda x: x.value_counts().index[0]).values

    df['UserRole'] = df['UserRole'].astype('category')
    df['OrganizationType'] = df['OrganizationType'].astype('category')

    # df_users['Score'] = users['Score'].mean().values

    # df_users = remove_outliers(df_users)

    return df_users

def add_cc(df):
    latitude = df.Latitude.values
    longitude = df.Longitude.values

    coordinates = []
    for lat, lng in zip(latitude, longitude):
        coordinates.append((lat, lng))

    countries = []
    results = rg.search(coordinates)
    for r in results:
        countries.append(r['cc'])

    df['Country'] = countries

    return df

def remove_outliers(df):
    outliers_lst  = []
    contin_cols = ['NumSims']
    for feature in contin_cols:
        Q1 = np.percentile(df.loc[:, feature], 25)
        Q3 = np.percentile(df.loc[:, feature], 75)

        # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
        step = 1.5 * (Q3 - Q1)

        # Find any points outside of Q1 - step and Q3 + step
        outliers_rows = df.loc[~((df[feature] >= Q1 - step) & (df[feature] <= Q3 + step)), :]

        outliers_lst.append(list(outliers_rows.index))

    outliers = list(itertools.chain.from_iterable(outliers_lst))
    # List of duplicate outliers
    dup_outliers = list(set([x for x in outliers if outliers.count(x) > 1]))
    # Remove duplicate outliers
    good_df = df.drop(df.index[dup_outliers]).reset_index(drop=True)

    return good_df

def get_fips_codes(df):
    df = df[df['Country'] == 'US']
    latitude = df.Latitude.values
    longitude = df.Longitude.values

    fips_codes = []
    for lat, lng in zip(latitude, longitude):
        url = 'http://data.fcc.gov/api/block/find?latitude={}&longitude={}&showall=true'.format(lat, lng)
        content = requests.get(url).content
        root = ET.fromstring(content)
        try:
            fips_codes.append(root[1].attrib['FIPS'])
        except:
            fips_codes.append('NA')

    df['FIPS'] = fips_codes

    return df

if __name__ == '__main__':
    # ---Create dataframes---
    df = read_data()
    df = add_cc(df)
    # scores = score_rows(df)
    # df['Score'] = scores
    df_users = create_user_df(df)
    df_users = add_cc(df_users)

    # ---Pickle dataframes---
    df.to_pickle('data/df.pkl')
    df_users.to_pickle('data/df_users.pkl')

    # ---Read back in pickled dataframes---
    # df = pd.read_pickle('data/new_full_df.pkl')
    # df_users = pd.read_pickle('data/new_user_df.pkl')

    # --Create USA dataframe with FIPS codes---
    # df_usa = get_fips_codes(df_users)
    # df_usa.to_pickle('data/usa_df.pkl')
    # df_usa = pd.read_pickle('data/usa_df.pkl')
