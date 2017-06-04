import numpy as np
import pandas as pd
from fancyimpute import KNN
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
import pdb

def read_data():
    df = pd.read_csv('combined_grid_and_run_data.csv', encoding='iso-8859-1')

    cols = ['Style',
            'MonthlyPurchaseCapacity',
            'UnreliableGrid',
            'UserCountry',
            'UserRole',
            'OrganizationType',
            'Sector',
            'Sample',
            'ProjectCategory',
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

    # create boolean columns of if a cost table has multiple lines
    cost_tables = ['Generator0CostTable', 'WindTurbine0CostTable', 'Battery0CostTable', 'Pv0CostTable', 'ConverterCostTable']
    costs = ['Gen', 'Wind', 'Bat', 'Pv', 'Con']
    for name, table in zip(costs, cost_tables):
        df[name+'CostMultiLines'] = df[table].astype(str).apply(lambda x: x.split('|')).apply(lambda x: len(x) > 1)

    # drop cost table columns no longer needed
    for table in cost_tables:
        df.drop(table, axis=1, inplace=True)

    # clean latitude and longitude columns
    lat_lon_cols = ['Latitude', 'Longitude']
    for col in lat_lon_cols:
        df[col] = df[col].replace(['0'], np.nan)
        df[col] = df[col].astype(float)

    # change type of User column
    df['User'] = df['User'].apply(pd.to_numeric, errors='coerce')
    # drop rows where User is null
    df = df[pd.notnull(df['User'])]
    # convert to int and then str types
    df['User'] = df['User'].astype(int)
    df['User'] = df['User'].astype(str)
    # drop any Users with IDs whose length is not 6
    df = df[df['User'].map(len) == 6]

    # change type of Created column to datetime
    df['Created'] = pd.to_datetime(df['Created'])

    # clean Sample column - if a sample is used = True, else = False
    df['Sample'] = np.where(df['Sample'].notnull(), True, False)

    # replace '0' with NaN in Style column
    df['Style'] = df['Style'].replace(['0'], np.nan)

    # replace '80' and '248' in 'UserCountry'
    df['UserCountry'] = df['UserCountry'].replace(['80', '248'], np.nan)

    # convert Sector to int type
    df['Sector'] = df['Sector'].astype(int)

    # drop rows where UserCountry is null
    # df = df[pd.notnull(df['UserCountry'])]

    # drop rows where Style is null
    # df = df[pd.notnull(df['Style'])]

    # clean 'UserRole'
    df['UserRole'] = df['UserRole'].replace(['Undergraduate Student', 'Post-graduate Student'], 'Student')
    df['UserRole'] = df['UserRole'].replace(['IT Professional'], 'IT Staff')
    df['UserRole'] = df['UserRole'].replace(['Tenured or Tenure-track Faculty'], 'Faculty')

    df['IsProUser'] = np.where(df['UserRole'].str.contains('Engineer|Regulator|Technician|Executive'), True, False)
    # encode string categorical data
    # enc_dct = dict()
    # encode_cols = ['Style', 'UserCountry', 'ProjectCategory']
    # for col in encode_cols:
    #     df[col] = df[col].apply(lambda x: str(x))
    #     le = LabelEncoder()
    #     le.fit(df[col])
    #     df[col] = le.transform(df[col])
    #     enc_dct[col] = le

    # dummize ProjectCategory and Style columns
    df = pd.get_dummies(df, prefix=['Project', 'Style'], prefix_sep='', columns=['ProjectCategory', 'Style'])

    df.dropna(axis=0, how='any', inplace=True)

    # convert boolean columns to int type
    bool_cols = ['MonthlyPurchaseCapacity', 'UnreliableGrid', 'Sample', 'Chp', 'GridConnected', 'GenCostMultiLines', 'WindCostMultiLines', 'BatCostMultiLines', 'PvCostMultiLines', 'ConCostMultiLines', 'IsProUser', 'ImportedWind', 'ImportedSolar']

    for col in bool_cols:
        if df[col].dtype != bool:
            df[col] = df[col].astype(bool)
        df[col] = df[col].astype(int)

    # create list of categorical and boolean columns that need to be imputed
    # impute_cat_cols = ['UserRole', 'OrganizationType', 'ImportedWind', 'ImportedSolar']
    # # sort columns by how many null values they have - column with fewest nulls is filled in first
    # sorted_cat_cols = percent_null(df, impute_cat_cols)
    #
    # # list of columns that do not have any null values
    # complete_cols = ['MonthlyPurchaseCapacity', 'Style', 'UserCountry', 'ProjectCategory', 'UnreliableGrid', 'Sector', 'LocationId', 'Sample', 'GridConnected', 'Gen_cost_multi_lines', 'Wind_cost_multi_lines', 'Bat_cost_multi_lines', 'Pv_cost_multi_lines', 'Con_cost_multi_lines']
    #
    # # impute values for discrete columns with missing values
    # clf_mod = RandomForestClassifier(n_jobs=-1)
    # impute_cat_rf(df, complete_cols, sorted_cat_cols, clf_mod, enc_dct)

    # create list of continuous columns that need to be imputed
    # impute_contin_cols = ['Latitude', 'Longitude']
    # # sort columns by how many null values they have - column with fewest nulls is filled in first
    # sorted_contin_cols = percent_null(df, impute_contin_cols)
    #
    # # list of columns that do not have any null values (updated with imputed discrete columns)
    # complete_cols = ['MonthlyPurchaseCapacity', 'Style', 'UserCountry', 'ProjectCategory', 'UserRole', 'OrganizationType', 'ImportedWind', 'ImportedSolar', 'UnreliableGrid', 'Sector', 'LocationId', 'Sample', 'GridConnected', 'Gen_cost_multi_lines', 'Wind_cost_multi_lines', 'Bat_cost_multi_lines', 'Pv_cost_multi_lines', 'Con_cost_multi_lines']
    #
    # # impute values for latitude and longitude
    # reg_mod = RandomForestRegressor(n_jobs=-1)
    # # mod = KNeighborsRegressor(n_neighbors=3)
    # impute_contin_rf(df, complete_cols, sorted_contin_cols, reg_mod)

    # reset dataframe index
    df.reset_index(drop=True, inplace=True)

    # inverse transform completed encoded columns back to strings/objects
    # for col, encoder in enc_dct.items():
    #     df[col] = encoder.inverse_transform(df[col])

    # convert completed bool columns back to bool type
    # bool_cols = ['MonthlyPurchaseCapacity', 'UnreliableGrid', 'Sample', 'Chp', 'GridConnected', 'ImportedWind', 'ImportedSolar', 'GenCostMultiLines', 'WindCostMultiLines', 'BatCostMultiLines', 'PvCostMultiLines', 'ConCostMultiLines', 'IsProUser']
    # for col in bool_cols:
    #     df[col] = df[col].astype(bool)

    return df #, enc_dct

def impute_cat_rf(df, complete_cols, cat_cols, model, dct):
    '''
    Imputes and fills null values in the passed in dataframe object according to the model specified. Also adds label encoder objects for each column to a passed in dictionary to the encoded values can be inversely transformed later.

    :param df: dataframe object
    :param complete_cols: list of column names for columsn in the dataframe that have no null values
    :param cat_cols: column names for which missing values will be imputed
    :param model: model to classify missing values
    :param dct: label encoder dictionary
    :returns: nothing
    '''
    complete = complete_cols
    for col in cat_cols:
        subset = df[col].values
        missing = pd.isnull(df[col]).values

        le = LabelEncoder()
        le.fit(subset[~missing])
        sub_enc = le.transform(subset[~missing])
        dct[col] = le
        subset[~missing] = sub_enc

        X = df.loc[:, [c for c in complete]].values
        # X_scaled = StandardScaler().fit_transform(X)

        model.fit(X[~missing], subset[~missing].astype('float64'))
        predictions = model.predict(X[missing])
        subset[missing] = predictions
        df[col] = subset
        df[col] = df[col].astype(int)
        complete.append(col)

def impute_contin_rf(df, complete_cols, contin_cols, model):
    '''
    Imputes and fills null values in the passed in dataframe object according to the model specified.

    :param df: dataframe object
    :param complete_cols: list of column names for columsn in the dataframe that have no null values
    :param discrete_cols: column names for which missing values will be imputed
    :param model: model to predict missing values
    :returns: nothing
    '''
    complete = complete_cols
    for col in contin_cols:
        subset = df[col].values
        X = df.loc[:, [c for c in complete]].values
        # X_scaled = StandardScaler().fit_transform(X)
        missing = np.isnan(subset)

        model.fit(X[~missing], subset[~missing])
        predictions = model.predict(X[missing])
        subset[missing] = predictions
        df[col] = subset
        complete.append(col)

def percent_null(df, cols):
    '''
    Calculates the percentage of null values in each each dataframe column and then orders the list of column names from least to greatest

    :param df: dataframe
    :param cols: list of column names to sort
    :returns: list of sorted column names
    '''
    col_lst = []
    for col in cols:
        num_null = sum(df[col].isnull())
        tot_num = len(df[col])
        percent_null = num_null / tot_num
        tup = (col, percent_null)
        col_lst.append(tup)

    lst_sorted = sorted(col_lst, key=(lambda x: x[1]), reverse=False)
    cols = [col for (col, precent) in lst_sorted]
    return cols

def kmodes_cluster():
    pass

def create_user_df(df):
    '''
    Create user dataframe from passed in dataframe, grouping by user. Takes the most frequent occuring value from users with multiple simulations.

    :param df: dataframe to build new dataframe
    :returns: created user dataframe
    '''
    # df.dropna(axis=0, how='any', inplace=True)

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
    cols = ['Style', # cat column
            'StyleExtension',
            'StyleRealtime',
            'StyleSimple',
            'StyleScheduled',
            'MonthlyPurchaseCapacity',
            'UnreliableGrid',
            'UserCountry', # cat column
            'UserRole', # cat column
            'OrganizationType', # cat column
            'Sector', # cat column (need to convert to categorical first)
            'Sample',
            'ProjectGrid',
            'ProjectIsland',
            'ProjectVillage',
            'Chp',
            'Latitude', # float - location
            'Longitude', # float - location
            'GridConnected',
            'ImportedWind',
            'ImportedSolar',
            'GenCostMultiLines',
            'WindCostMultiLines',
            'BatCostMultiLines',
            'PvCostMultiLines',
            'ConCostMultiLines',
            'IsProUser']

    for col in cols:
        df_users[col] = users[col].agg(lambda x: x.value_counts().index[0]).values

    # create column of the time (in days) between first and last simulation for each user
    diff_first_last = df.groupby('User')['Created'].apply(lambda x: x.max() - x.min()).values
    df_users['DaysSinceFirst'] = diff_first_last.astype('timedelta64[D]').astype(int)

    return df_users


if __name__ == '__main__':
    df = read_data()
    df_users = create_user_df(df)
