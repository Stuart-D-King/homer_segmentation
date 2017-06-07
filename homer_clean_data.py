import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from geopy.geocoders import Nominatim, GoogleV3, NaviData
import pdb
import itertools

def read_data():
    df = pd.read_csv('data/combined_grid_and_run_data.csv', encoding='iso-8859-1')

    cols = ['Style',
            'MonthlyPurchaseCapacity',
            'UnreliableGrid',
            'UserRole',
            'OrganizationType',
            'Sector',
            'Sample',
            'ProjectCategory',
            #'ProjectCountry',
            'Chp',
            'Latitude',
            'Longitude',
            'Created',
            'User',
            'GridConnected',
            'Generator0CostTable',
            'WindTurbine0CostTable',
            'Battery0CostTable',
            'Pv0CostTable',
            'ConverterCostTable',
            'ImportedWind',
            'ImportedSolar']

    df = df[cols]

    print('Cleaning columns...')
    # clean 'Style'
    df['Style'] = df['Style'].replace(['0', np.nan], 'NA')

    # clean 'UserRole'
    df['UserRole'] = df['UserRole'].replace(['Undergraduate Student', 'Post-graduate Student'], 'Student')
    df['UserRole'] = df['UserRole'].replace(['IT Professional'], 'IT Staff')
    df['UserRole'] = df['UserRole'].replace(['Tenured or Tenure-track Faculty'], 'Faculty')
    df['UserRole'] = df['UserRole'].replace([np.nan, 'Other'], 'NA')

    # clean 'OrganizaitonType'
    df['OrganizationType'] = df['OrganizationType'].replace([np.nan, 'Other'], 'NA')

    # clean 'Sample'; if a sample is used = True, else = False
    df['Sample'] = np.where(df['Sample'].notnull(), True, False)

    # clean 'ProjectCategory'
    df['ProjectCategory'] = df['ProjectCategory'].replace(['0', np.nan], 'NA')

    # clean 'ProjectCountry''
    # df['ProjectCountry'] = df['ProjectCountry'].astype(str)
    # df['ProjectCountry'] = df['ProjectCountry'].apply(lambda x: x.encode('ascii', 'ignore'))
    # df['ProjectCountry'] = df['ProjectCountry'].apply(lambda x: x.decode('utf-8'))
    #
    # df['ProjectCountry'] = df['ProjectCountry'].apply(lambda x: 'saudi arabia' if 'saudi' in x else x)
    # df['ProjectCountry'] = df['ProjectCountry'].apply(lambda x: 'uae' if 'united arab' in x else x)
    # df['ProjectCountry'] = df['ProjectCountry'].apply(lambda x: 'usa' if 'united states' in x else x)
    # df['ProjectCountry'] = df['ProjectCountry'].apply(lambda x: np.nan if '?' in x else x.lower())
    # df['ProjectCountry'] = df['ProjectCountry'].replace(['nan'], 'NA')

    # clean latitude and longitude columns
    # will look into 'geopy' to see if I can impute coordinates for simulaitons without latitude and longitude
    lat_lon_cols = ['Latitude', 'Longitude']
    for col in lat_lon_cols:
        df[col] = df[col].replace(['0'], np.nan)
        df[col] = df[col].astype(float)

    # change type of 'Created' to datetime
    df['Created'] = pd.to_datetime(df['Created'])

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

    # create 'IsProUser'; True if an Engineer, Regulator, Technician, or Executive, False if none of those
    df['IsProUser'] = np.where(df['UserRole'].str.contains('Engineer|Regulator|Technician|Executive'), True, False)

    # create 'AcademicOrIndividual'; True if from an academic institution or an interested individual
    df['AcademicOrIndividual'] = np.where(df['OrganizationType'].str.contains('Academic|Individual|NA'), True, False)

    # dummize 'ProjectCategory' and 'Style' columns
    df_dummies = pd.get_dummies(df, prefix=['Project', 'Style'], prefix_sep='', columns=['ProjectCategory', 'Style'])
    df = pd.concat([df[['Style', 'ProjectCategory']], df_dummies], axis=1)

    # print('Cleaning country column...')
    # latitude = df.Latitude.values
    # longitude = df.Longitude.values
    #
    # countries = []
    # geolocator = Nominatim()
    # geolocator = GoogleV3()
    # for lat, lng in zip(latitude, longitude):
    #     coord = '{}, {}'.format(str(lat).strip(), str(lng).strip())
    #     location = geolocator.reverse(coord)
    #     country = location.raw['address']['country']
    #     countries.append(country)

    # df['Country'] = countries

    df.dropna(axis=0, how='any', inplace=True)

    print('Converting booleans to integers...')
    # convert boolean columns to int type
    bool_cols = ['MonthlyPurchaseCapacity', 'UnreliableGrid', 'Sample', 'Chp', 'GridConnected', 'ImportedWind', 'ImportedSolar', 'GenCostMultiLines', 'WindCostMultiLines', 'BatCostMultiLines', 'PvCostMultiLines', 'ConCostMultiLines', 'IsProUser', 'AcademicOrIndividual']

    for col in bool_cols:
        if df[col].dtype != bool:
            df[col] = df[col].astype(bool)
        df[col] = df[col].astype(int)

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
    # get the total simulations for each user
    counts = users.size()
    user_id = counts.index.values
    num_sims = counts.values

    # create dataframe of user ids and add number of simulations column
    df_users = pd.DataFrame(data=user_id, columns=['UserId'])
    df_users['NumSims'] = num_sims

    # using the mode for each column from the full data frame, add column values to each new column created in user dataframe
    cols = ['Style',
            'StyleExtension',
            'StyleRealtime',
            'StyleSimple',
            'StyleScheduled',
            'MonthlyPurchaseCapacity',
            'UnreliableGrid',
            'UserRole',
            'OrganizationType',
            'Sector',
            'Sample',
            'ProjectCategory',
            'ProjectGrid',
            'ProjectIsland',
            'ProjectVillage',
            # 'ProjectCountry',
            'Chp',
            'Latitude',
            'Longitude',
            'GridConnected',
            'ImportedWind',
            'ImportedSolar',
            'GenCostMultiLines',
            'WindCostMultiLines',
            'BatCostMultiLines',
            'PvCostMultiLines',
            'ConCostMultiLines',
            'IsProUser',
            'AcademicOrIndividual']

    for col in cols:
        df_users[col] = users[col].agg(lambda x: x.value_counts().index[0]).values

    # create column of the time (in days) between first and last simulation for each user
    diff_first_last = df.groupby('User')['Created'].apply(lambda x: x.max() - x.min()).values
    df_users['DaysSinceFirst'] = diff_first_last.astype('timedelta64[D]').astype(int)

    df_users = remove_outliers(df_users)

    return df_users

def remove_outliers(df):
    outliers_lst  = []
    contin_cols = ['NumSims', 'DaysSinceFirst']
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

def one_hot(main_df):
    df = main_df.copy()

    # encode string categorical data
    enc_dct = dict()
    encode_cols = ['Style', 'ProjectCategory', 'UserRole', 'OrganizationType']
    for col in encode_cols:
        le = LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        enc_dct[col] = le

    cols = [
            # 'Style',
            'ProjectCategory',
            'UserRole',
            'OrganizationType',
            'Sector',
            # 'MonthlyPurchaseCapacity',
            # 'UnreliableGrid',
            'Sample',
            # 'Chp',
            # 'GridConnected',
            'ImportedWind',
            'ImportedSolar',
            'GenCostMultiLines',
            'WindCostMultiLines',
            'BatCostMultiLines',
            'PvCostMultiLines',
            'ConCostMultiLines',
            # 'IsProUser',
            # 'AcademicOrIndividual'
            ]

    df = df[cols]
    X = df.values
    ohe = OneHotEncoder()
    X = ohe.fit_transform(X)
    return X

def get_fips_codes(df):
    df = df[df['Country'] == 'United States of America']
    latitude = df.Latitude.values
    longitude = df.Longitude.values

    fips_codes = []
    for lat, lng in zip(latitude, longitude):
        url = 'http://data.fcc.gov/api/block/find?latitude={}&longitude={}&showall=true'.format(lat), lng)
        content = requests.get(url).content
        root = ET.fromstring(content)
        try:
            fips_codes.append(root[1].attrib['FIPS'])
        except:
            fips_codes.append('NA')

    df['FIPS'] = fips_codes

if __name__ == '__main__':
    # ---Create dataframes---
    # df = read_data()
    # df_users = create_user_df(df)
    # users_ohe = one_hot(df_users)

    # ---Pickle dataframes---
    # df.to_pickle('data/full_df.pkl')
    # df_users.to_pickle('data/user_df.pkl')
    # np.savetxt('data/users_ohe.csv', users_ohe.todense())

    # ---Read back in pickled dataframes---
    # df = pd.read_pickle('data/full_df.pkl')
    # df_users=pd.read_pickle('data/user_df.pkl')
    # users_ohe = np.loadtxt('data/users_ohe.csv')
