import pytablewriter
import pandas as pd
import numpy as np

def user_counts(df):
    grps = df.UserRole.value_counts(dropna=False)
    pct = [x / float(sum(grps)) for x in grps]
    s1 = pd.Series(grps.values, index=grps.index, name='Count')
    s2 = pd.Series(pct, index=grps.index, name='Total%')
    s2 = s2.apply(lambda x: round(x*100, 2))
    df_grp = pd.concat([s1, s2], axis=1)
    return df_grp

def org_counts(df):
    grps = df.OrganizationType.value_counts(dropna=False)
    pct = [x / float(sum(grps)) for x in grps]

    s1 = pd.Series(grps.values, index=grps.index, name='Count')
    s2 = pd.Series(pct, index=grps.index, name='Total%')
    s2 = s2.apply(lambda x: round(x*100, 2))
    df_grp = pd.concat([s1, s2], axis=1)
    return df_grp

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
    return df

def usersims_by_cluster(df):
    pt = pd.pivot_table(df, values=['Created'], index=['User'], columns=['Cluster'], aggfunc='count', fill_value=0)

    cluster_df = pd.DataFrame(pt.iloc[:, 0].index)
    cluster_df['Cluster 1'] = pt.iloc[:, 0].values
    cluster_df['Cluster 2'] = pt.iloc[:, 1].values
    cluster_df['Cluster 3'] = pt.iloc[:, 2].values
    cluster_df['Cluster 4'] = pt.iloc[:, 3].values

    output = round(cluster_df.describe().loc[['mean', 'std', 'max']], 2)
    return output

def write_mdtable(df):
    writer = pytablewriter.MarkdownTableWriter()
    writer.table_name = "example_table"

    writer.header_list = ['Index', 'Cluster 1: Count', 'Cluster 1: Total%', 'Cluster 2: Count', 'Cluster 2: Total%', 'Cluster 3: Count', 'Cluster 3: Total%', 'Cluster 4: Count', 'Cluster 4: Total%']
    df.reset_index(level=0, inplace=True)
    # writer.header_list = df.columns.tolist()
    writer.value_matrix = df.values.tolist()
    print(writer.write_table())


if __name__ == '__main__':
    df = pd.read_pickle('data/df_clustered.pkl')

    C1, C2, C3, C4 = df[df['Cluster'] == 1], df[df['Cluster'] == 2], df[df['Cluster'] == 3], df[df['Cluster'] == 4]

    # user = user_counts(df)
    # org = org_counts(df)
    search_gen = search_space_counts('MultiGenSearch')
    search_pv = search_space_counts('MultiPvSearch')
    search_bat = search_space_counts('MultiBatSearch')
    search_con = search_space_counts('MultiConSearch')
    search_wind = search_space_counts('MultiWindSearch')

    # sims_by_cluster = usersims_by_cluster(df)
