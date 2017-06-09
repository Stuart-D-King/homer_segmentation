import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pdb
import itertools
import requests
import reverse_geocoder as rg
import xml.etree.ElementTree as ET
from collections import defaultdict

def read_data():
    df = pd.read_csv('data/combined_grid_and_run_data.csv', encoding='iso-8859-1')

    cols = ['UserRole',
            'OrganizationType',
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
            'ImportedSolar',
            'Electric1Peak',
            'Generator0Capital',
            'Battery0Capital',
            'WindTurbine0Capital',
            'Pv0Capital',
            'Generator0'
            ]

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

    # set 'OrganizationType' as category type
    df['OrganizationType'] = df['OrganizationType'].astype('category')

    print('Creating new columns...')
    # create new 'ElectDefault' column
    df['ElectricNotDefault'] = df['Electric1Peak'].apply(lambda x: isinstance(x, float) and x != 0 and x < 1000000)

    # create new 'GeneratorDefault' column
    df['GeneratorNotDefault'] = np.where(df['Generator0'] == 'Autosize Genset', False, True)

    # create new capital cost columns to see if a user input a value or not
    df['GenCapCost'] = np.where(df['Generator0Capital'] != 0, True, False)
    df['BatCapCost'] = np.where(df['Battery0Capital'] != 0, True, False)
    df['WindCapCost'] = np.where(df['WindTurbine0Capital'] != 0, True, False)
    df['PvCapCost'] = np.where(df['Pv0Capital'] != 0, True, False)

    # drop columns no longer needed
    for table in ['Electric1Peak', 'Generator0', 'Generator0Capital', 'Battery0Capital', 'WindTurbine0Capital', 'Pv0Capital']:
        df.drop(table, axis=1, inplace=True)

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
    bool_cols = ['ImportedWind', 'ImportedSolar', 'GenCostMultiLines', 'WindCostMultiLines', 'BatCostMultiLines', 'PvCostMultiLines', 'ConCostMultiLines', 'ElectricNotDefault', 'GeneratorNotDefault', 'GenCapCost', 'BatCapCost', 'WindCapCost', 'PvCapCost']

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
    # df.dropna(axis=0, how='any', inplace=True)

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
            'Latitude',
            'Longitude',
            'ImportedWind',
            'ImportedSolar',
            'ElectricNotDefault',
            'GeneratorNotDefault',
            'GenCapCost',
            'BatCapCost',
            'WindCapCost',
            'PvCapCost',
            'GenCostMultiLines',
            'WindCostMultiLines',
            'BatCostMultiLines',
            'PvCostMultiLines',
            'ConCostMultiLines'
            ]

    for col in cols:
        df_users[col] = users[col].agg(lambda x: x.value_counts().index[0]).values

    # change dtype of 'UserRole' and 'OrganizationType' to category
    df_users['UserRole'] = df_users['UserRole'].astype('category')
    df_users['OrganizationType'] = df_users['OrganizationType'].astype('category')

    # to cacluate the number of changed inputs
    input_cols = ['ImportedWind',
                'ImportedSolar',
                'ElectricNotDefault',
                'GeneratorNotDefault',
                'GenCapCost',
                'BatCapCost',
                'WindCapCost',
                'PvCapCost',
                'GenCostMultiLines',
                'WindCostMultiLines',
                'BatCostMultiLines',
                'PvCostMultiLines',
                'ConCostMultiLines'
                ]

    # create 'NumChangedInputs' column = sums all true values in boolean columns
    df_users['NumChangedInputs'] = df_users[input_cols].sum(axis=1)
    # df_users['Score'] = users['Score'].mean().values

    # remove outliers and reset index in 'NumSims' column
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

def remove_outliers(df_):
    df = df_.copy()

    Q1 = np.percentile(df.loc[:, 'NumSims'], 25)
    Q3 = np.percentile(df.loc[:, 'NumSims'], 75)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)

    # Find any points outside of Q1 - step and Q3 + step
    outliers_rows = df.loc[~((df['NumSims'] >= Q1 - step) & (df['NumSims'] <= Q3 + step)), :]

    clean_df = df.drop(df.index[outliers_rows.index.tolist()]).reset_index(drop=True)

    return clean_df

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
            fips_codes.append(0)

    df['FIPS'] = fips_codes

    df.reset_index(drop=True, inplace=True)
    # df = remove_outliers(df)

    return df

if __name__ == '__main__':
    # ---Create dataframes---
    # df = read_data()
    # df = add_cc(df)
    # df['Score'] = score_rows(df)
    # df_users = create_user_df(df)
    # df_users = add_cc(df_users)

    # ---Pickle dataframes---
    # df.to_pickle('data/df.pkl')
    # df_users.to_pickle('data/df_users.pkl')

    # ---Read back in pickled dataframes---
    # df = pd.read_pickle('data/df.pkl')
    # df_users = pd.read_pickle('data/df_users.pkl')

    # AFTER CLUSTERING
    # ---Read in clustered dataframes---
    df_clustered = pd.read_pickle('data/df_clustered.pkl')
    df_users_clustered = pd.read_pickle('data/df_users_clustered.pkl')

    # ---Create USA dataframe with FIPS codes---
    df_usa = get_fips_codes(df_clustered)
    df_users_usa = get_fips_codes(df_users_clustered)

    # ---Pickle USA dataframes---
    df_usa.to_pickle('data/df_usa.pkl')
    df_users_usa.to_pickle('data/df_users_usa')

    # ---Read back in pickled USA dataframes---
    # df_usa = pd.read_pickle('data/df_usa.pkl')
    # df_users_usa = pd.read_pickle('data/df_users_usa.pkl')

    # ---Create user and full dataframe country dictionaries---
    # country_dct = defaultdict(list)
    # for idx, country in enumerate(df_clustered['Country']):
    #     country_dct[country].append(idx)

    # user_country_dct = defaultdict(list)
    # for idx, country in enumerate(df_users_clustered['Country']):
    #     user_country_dct[country].append(idx)
    #
