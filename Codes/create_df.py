import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from fiona.crs import from_epsg


# function for long separating lines
def l():
    print('-' * 100)


# function for short separating lines
def s():
    print("-" * 40)


def import_csv_file(step):
    print("Importing parameters and input files..")
    os.chdir(r'Input//')
    config = pd.read_csv('Configuration.csv').values
    input_csv = config[0, 1]
    input_sub = config[1, 1]
    crs = config[2, 1]
    resolution = float(config[3, 1])
    unit = config[4, 1]
    pop_load = float(config[5, 1])
    pop_thresh = float(config[6, 1])
    line_bc = float(config[7, 1])
    limit_hv = float(config[8, 1])
    limit_mv = float(config[9, 1])
    if step == 1:
        df = pd.read_csv(input_csv + '.csv', sep=',')
        print("Input files successfully imported.")
        os.chdir(r'..//')
        return df, input_sub, input_csv, crs, resolution, unit, pop_load, \
            pop_thresh, line_bc, limit_hv, limit_mv
    elif step == 2 or step == 3:
        os.chdir(r'..//')
        os.chdir(r'Output//Datasets')
        df_weighted = pd.read_csv(input_csv + '_weighted.csv')
        valid_fields = ['ID', 'X', 'Y', 'Population', 'Elevation', 'Weight']
        blacklist = []
        for x in df_weighted.columns:
            if x not in valid_fields:
                blacklist.append(x)
        df_weighted.drop(blacklist, axis=1, inplace=True)
        df_weighted.drop_duplicates(['ID'], keep='last', inplace=True)
        print("Input files successfully imported.")
        os.chdir(r'..//..')
        if step == 3:
            l()
            print("Importing Clusters..")
            os.chdir(r'Output//Clusters')
            geo_df_clustered = gpd.read_file("geo_df_clustered.shp")
            clusters_list = np.unique(geo_df_clustered['clusters'])
            clusters_list = np.delete(clusters_list, np.where(
                clusters_list == -1))  # removes the noises
            clusters_list = np.tile(clusters_list, (3, 1))
            for i in clusters_list[0]:
                pop_i = sum(geo_df_clustered.loc
                            [geo_df_clustered['clusters'] == i, 'Population'])
                clusters_list[1, np.where(clusters_list[0] == i)] = pop_i
                clusters_list[
                    2, np.where(clusters_list[0] == i)] = pop_i * pop_load

            print("Clusters successfully imported")
            l()
            os.chdir(r'..//..')
            return df_weighted, input_sub, input_csv, crs, resolution, unit, \
                pop_load, pop_thresh, line_bc, limit_hv, limit_mv, \
                geo_df_clustered, clusters_list
        return df_weighted, input_sub, input_csv, crs, resolution, unit, \
            pop_load, pop_thresh, line_bc, limit_hv, limit_mv


def weighting(df):
    df_weighted = df.dropna(subset=['Elevation'])
    df_weighted.reset_index(drop=True)
    df_weighted.Slope.fillna(value=0, inplace=True)
    df_weighted.Land_cover.fillna(method='bfill', inplace=True)
    df_weighted['Land_cover'] = df_weighted['Land_cover'].round(0)
    df_weighted.Population.fillna(value=0, inplace=True)
    df_weighted['Weight'] = 0
    print('Weighting the Dataframe..')
    os.chdir(r'Input//')
    landcover_csv = pd.read_csv('Landcover.csv')
    os.chdir(r'..//')
    del df
    # Weighting section
    for index, row in df_weighted.iterrows():
        # Slope conditions
        df_weighted.loc[index, 'Weight'] = row.Weight + math.exp(
            0.01732867951 * row.Slope)
        # Land cover
        df_weighted.loc[index, 'Weight'] += landcover_csv.WeightGLC[
            landcover_csv.iloc[:, 0] == row.Land_cover].values[0]
        # Road distance conditions
        if row.Road_dist < 100:
            df_weighted.loc[index, 'Weight'] = 1
        elif row.Road_dist < 1000:
            df_weighted.loc[index, 'Weight'] += 5 * row.Road_dist / 1000
        else:
            df_weighted.loc[index, 'Weight'] += 5

    df_weighted.drop_duplicates(['ID'], keep='last', inplace=True)
    df_weighted = df_weighted.drop(
        ['Slope', 'Land_cover', 'River_flow', 'NAME', 'Road_dist'],
        axis=1)
    # another cleaning
    valid_fields = ['ID', 'X', 'Y', 'Population', 'Elevation', 'Weight']
    blacklist = []
    for x in df_weighted.columns:
        if x not in valid_fields:
            blacklist.append(x)
    df_weighted.drop(blacklist, axis=1, inplace=True)
    df_weighted.drop_duplicates(['ID'], keep='last', inplace=True)
    print("Cleaning and weighting process completed")
    s()
    return df_weighted


def creating_geodataframe(df_weighted, crs, unit, input_csv, step):
    print("Creating the GeoDataFrame..")
    geometry = [Point(xy) for xy in zip(df_weighted['X'], df_weighted['Y'])]
    geo_df = gpd.GeoDataFrame(df_weighted, geometry=geometry,
                              crs=from_epsg(crs))
    del df_weighted
    # - Check crs conformity for the clustering procedure
    if unit == '0':
        print('Attention: the clustering procedure requires measurements in '
              'meters. Therefore, your dataset must be reprojected.')
        crs = int(input("Provide a valid crs code to which reproject:"))
        geo_df['geometry'] = geo_df['geometry'].to_crs(epsg=crs)
        print("Done")
    print("GeoDataFrame ready!")
    s()
    if step == 1:
        os.chdir(r'Output//Datasets')
        geo_df.to_csv(input_csv + "_weighted.csv")
        geo_df.to_file(input_csv + "_weighted.shp")
        os.chdir(r'..//..')
        print("GeoDataFrame exported. You can find it in the output folder.")
        l()

    print("In the considered area there are " + str(
        len(geo_df)) + " points, for a total of " +
          str(int(geo_df['Population'].sum(axis=0))) +
          " people which could gain access to electricity.")

    d = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
    pop_points = pd.DataFrame(data=d).values

    return geo_df, pop_points
