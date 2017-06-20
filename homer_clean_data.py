from collections import defaultdict
import pdb
import numpy as np
import pandas as pd
import requests
import reverse_geocoder as rg
import xml.etree.ElementTree as ET


def read_data():
    '''
    Read in, clean, and prepare data for precessing.

    :param: None
    :returns: cleaned and prepared dataframe
    '''
    # df = pd.read_csv('data/combined_grid_and_run_data.csv', encoding='iso-8859-1')
    df = pd.read_csv('data/all_data.csv')
    cols = ['UserRole',
            'OrganizationType',
            'Latitude',
            'Longitude',
            'User',
            'Created',
            'Sample',
            'Generator0',
            'Generator0SearchSpace',
            'WindTurbine0',
            'WindTurbine0SearchSpace',
            'Battery0',
            'Battery0SearchSpace',
            'Pv0',
            'Pv0SearchSpace',
            'Converter',
            'ConverterSearchSpace',
            'ImportedWind',
            'ImportedSolar'
            ]
    df = df[cols]

    print('Cleaning columns...')
    # change type of 'Created' to datetime
    df['Created'] = pd.to_datetime(df['Created'])
    df = df[df['Created'].dt.year > 2004]
    # clean 'UserRole'
    df['UserRole'] = df['UserRole'].replace(['Student', 'Undergraduate Student', 'Post-graduate Student', 'Tenured or Tenure-track Faculty', 'Faculty', 'Research Staff'], 'Academic')
    df['UserRole'] = df['UserRole'].replace(['Engineer', 'Mechanic/Technician/Facility Manager'], 'Technical')
    df['UserRole'] = df['UserRole'].replace(['IT Professional', 'IT Staff', 'Sales/Marketing', 'Purchasing Agent', 'Executive', 'Planner/Regulator/Policy Maker'], 'Business')
    df['UserRole'] = df['UserRole'].replace([np.nan, 'Personal Interest', 'Staff', 'Other'], 'NA')
    # clean 'OrganizaitonType'
    df['OrganizationType'] = df['OrganizationType'].replace([np.nan, 'Other', 'Interested Individual', 'Microgrid End User (all types)'], 'NA')
    df['OrganizationType'] = df['OrganizationType'].replace(['Engineering Services Company', 'Electric Distribution Utility', 'Independent Power Producer', 'Project Developer'], 'Engineering')
    df['OrganizationType'] = df['OrganizationType'].replace(['EPC/Construction Company', 'Solar Installation Company', 'Equipment Vendor'], 'Vendor')
    df['OrganizationType'] = df['OrganizationType'].replace(['Government', 'Non-Governmental Organization (NGO)'], 'Public')
    df['OrganizationType'] = df['OrganizationType'].replace(['Academic Institution or Research Center'], 'Academic')
    df['OrganizationType'] = df['OrganizationType'].replace(['Other Professional Services Company', 'Finance Organization'], 'Service')

    print('Creating new columns...')

    def f(x):
        '''
        Convert categorical variable to numeric: 'NA' --> -1, 'False' --> 0, 'True' --> 1
        '''
        if x == 'NA':
            val = -1
        elif x == 'False':
            val = 0
        else:
            val = 1
        return val

    # Create multiple generator search variable
    df['MultiGenSearch'] = np.where(df['Generator0'].notnull(), df['Generator0SearchSpace'].astype(str).apply(lambda x: x.split('|')).apply(lambda x: len(x) > 2), 'NA')
    # Create multiple turbine search variable
    df['MultiWindSearch'] = np.where(df['WindTurbine0'].notnull(), df['WindTurbine0SearchSpace'].astype(str).apply(lambda x: x.split('|')).apply(lambda x: len(x) > 2), 'NA')
    # Create multiple battery search variable
    df['MultiBatSearch'] = np.where(df['Battery0'].notnull(), df['Battery0SearchSpace'].astype(str).apply(lambda x: x.split('|')).apply(lambda x: len(x) > 2), 'NA')
    # Create multiple solar panel search variable
    df['MultiPvSearch'] = np.where(df['Pv0'].notnull(), df['Pv0SearchSpace'].astype(str).apply(lambda x: x.split('|')).apply(lambda x: len(x) > 2), 'NA')
    # Create multiple converter search variable
    df['MultiConSearch'] = np.where(df['Converter'].notnull(), df['ConverterSearchSpace'].astype(str).apply(lambda x: x.split('|')).apply(lambda x: len(x) > 2), 'NA')
    # create new 'GeneratorDefault' variable
    df['DefaultGenerator'] = np.where(df['Generator0'].notnull(), np.where(df['Generator0'] == 'Autosize Genset', True, False), 'NA')
    # clean 'Sample'; if a sample is used = True, else = False
    df['Sample'] = np.where(df['Sample'].notnull(), True, False)

    # drop columns no longer needed
    for col in ['Generator0', 'Generator0SearchSpace', 'WindTurbine0', 'WindTurbine0SearchSpace', 'Battery0', 'Battery0SearchSpace', 'Pv0', 'Pv0SearchSpace', 'Converter', 'ConverterSearchSpace']:
        df.drop(col, axis=1, inplace=True)

    # clean latitude and longitude columns
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

    print('Converting booleans to integers...')
    # convert boolean columns to int type
    bool_cols = ['ImportedWind', 'ImportedSolar', 'Sample']

    for col in bool_cols:
        if df[col].dtype != bool:
            df[col] = df[col].astype(bool)
        df[col] = df[col].astype(int)

    # drop nulls
    df.dropna(axis=0, how='any', inplace=True)
    # reset dataframe index
    df.reset_index(drop=True, inplace=True)

    print('All done!')
    return df


def create_user_df(df):
    '''
    Create user dataframe from passed in dataframe, grouping by user. Takes the most frequent occuring value from users with multiple simulations.

    :param df: dataframe to build new dataframe
    :returns: created user dataframe
    '''
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

    for col in cols:
        df_users[col] = users[col].agg(lambda x: x.value_counts().index[0]).values

    # remove outliers and reset index in 'NumSims' column
    # df_users = remove_outliers(df_users)

    return df_users


def add_country_codes(df):
    '''
    Use latitude and longitude coordinates to determine the country of a project.

    :param df: dataframe with coordinates
    :returns: dataframe with project country codes appended
    '''
    latitude = df.Latitude.values
    longitude = df.Longitude.values
    coordinates = []

    for lat, lng in zip(latitude, longitude):
        coordinates.append((lat, lng))

    codes = []
    results = rg.search(coordinates)

    for r in results:
        try:
            code = r['cc']
            codes.append(code)
        except:
            codes.append('NA')

    df['Country'] = codes

    return df


def remove_outliers(df_):
    '''
    Remove outliers that are 1.5 times above and below the 75th and 25th percentiles.

    :param df_: dataframe from which outliers are to be removed
    :returns: cleaned dataframe
    '''
    df = df_.copy()
    Q1 = np.percentile(df.loc[:, 'NumSims'], 25)
    Q3 = np.percentile(df.loc[:, 'NumSims'], 75)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    # Find any points outside of Q1 - step and Q3 + step
    outliers_rows = df.loc[~((df['NumSims'] >= Q1 - step) & (df['NumSims'] <= Q3 + step)), :]
    clean_df = df.drop(df.index[outliers_rows.index.tolist()]).reset_index(drop=True)

    return clean_df


def fips_codes(df):
    '''
    Add FIPS codes for U.S. simulations by pinging FCC API.

    :param df: dataframe for which FIPS codes are to be determined
    :returns: dataframe of U.S. simulations with FIPS codes appended
    '''
    df = df[df['Country'] == 'US']
    latitude = df.Latitude.values
    longitude = df.Longitude.values
    coordinates = [(lat, lng) for lat, lng in zip(latitude, longitude)]
    unique = list(set(coordinates))
    fips_codes = []

    for lat, lng in unique:
        try:
            url = 'http://data.fcc.gov/api/block/find?latitude={}&longitude={}&showall=true'.format(lat, lng)
            content = requests.get(url).content
            root = ET.fromstring(content)
            fips_codes.append(root[1].attrib['FIPS'])
            print(root[1].attrib['FIPS'])
        except:
            fips_codes.append(0)

    dct = dict(zip(unique, fips_codes))
    df['lat_lng'] = list(zip(latitude, longitude))
    df['FIPS'] = df['lat_lng'].apply(lambda x: dct[x])
    df.drop('lat_lng', axis=1, inplace=True)

    return df


if __name__ == '__main__':
    # ---Create dataframes---
    # df = read_data()
    # df = add_country_codes(df)
    # df_users = create_user_df(df)
    # df_users = add_country_codes(df_users)

    # ---Pickle dataframes---
    # df.to_pickle('data/df.pkl')
    # df_users.to_pickle('data/df_users.pkl')

    # ---Read back in pickled dataframes---
    # df = pd.read_pickle('data/df.pkl')
    # df_users = pd.read_pickle('data/df_users.pkl')

    # AFTER CLUSTERING
    # ---Read in clustered dataframes---
    df_clustered = pd.read_pickle('data/df_clustered.pkl')
    # df_users_clustered = pd.read_pickle('data/df_users_clustered.pkl')

    # ---Create USA dataframe with FIPS codes---
    # df_usa = fips_codes(df_clustered)
    # df_users_usa = fips_codes(df_users_clustered)

    # ---Pickle USA dataframes---
    # df_usa.to_pickle('data/df_usa.pkl')
    # df_users_usa.to_pickle('data/df_users_usa.pkl')

    # ---Read back in pickled USA dataframes---
    df_usa = pd.read_pickle('data/df_usa.pkl')
    # df_users_usa = pd.read_pickle('data/df_users_usa.pkl')

    # ---Create user and full dataframe country dictionaries---
    # country_dct = defaultdict(list)
    # for idx, country in enumerate(df_clustered['Country']):
    #     country_dct[country].append(idx)

    # user_country_dct = defaultdict(list)
    # for idx, country in enumerate(df_users_clustered['Country']):
    #     user_country_dct[country].append(idx)
