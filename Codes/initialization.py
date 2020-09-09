"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Initialization Code

Code for importing input GIS files, perform the weighting strategy and creating
the initial Point geodataframe.
"""

import os
import math
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from supporting_GISEle2 import l, s
from Codes import collecting, processing


def import_csv_file(step):
    """
    Imports all the parameters and information given by the user in the
    Configuration.csv file. If starting from further steps than the first,
    it reads and returns the output of previous steps.
    :param step: From which part of GISEle the user is starting
    :return: All parameters contained in the Configuration.csv
    """
    print("Importing parameters and input files..")
    os.chdir(r'Input//')
    config = pd.read_csv('Configuration.csv').values
    data_import = config[0, 1]
    input_csv = config[1, 1]
    input_sub = config[2, 1]
    crs = int(config[3, 1])
    resolution = float(config[4, 1])
    unit = config[5, 1]
    pop_load = float(config[6, 1])
    pop_thresh = float(config[7, 1])
    line_bc = float(config[8, 1])
    sub_cost_hv = float(config[9, 1])
    sub_cost_mv = float(config[10, 1])
    branch = config[11, 1]
    pop_thresh_lr = float(config[12, 1])
    line_bc_col = float(config[13, 1])
    full_ele = config[14, 1]
    wt = config[15, 1]
    coe = float(config[16, 1])
    grid_ir = float(config[17, 1])
    grid_om = float(config[18, 1])
    grid_lifetime = int(config[19, 1])

    if step == 1:
        if data_import == 'yes':
            os.chdir('..')
            study_area = collecting.data_gathering(crs)
            df = processing.create_mesh(study_area, crs, resolution)
            return df, input_sub, input_csv, crs, resolution, unit, pop_load, \
                pop_thresh, line_bc, sub_cost_hv, sub_cost_mv, branch, \
                pop_thresh_lr, line_bc_col, full_ele, wt, coe, grid_ir, \
                grid_om, grid_lifetime

        df = pd.read_csv(input_csv + '.csv', sep=',')
        print("Input files successfully imported.")
        os.chdir(r'..//')
        return df, input_sub, input_csv, crs, resolution, unit, pop_load, \
            pop_thresh, line_bc, sub_cost_hv, sub_cost_mv, branch, \
            pop_thresh_lr, line_bc_col, full_ele, wt, coe, grid_ir, \
            grid_om, grid_lifetime
    elif step > 1:
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
        if step == 2:
            return df_weighted, input_sub, input_csv, crs, resolution, unit, \
                   pop_load, pop_thresh, line_bc, sub_cost_hv, sub_cost_mv, \
                   branch, pop_thresh_lr, line_bc_col, full_ele, wt, \
                   coe, grid_ir, grid_om, grid_lifetime
        l()
        print("Importing Clusters..")
        os.chdir(r'Output//Clusters')
        geo_df_clustered = gpd.read_file("geo_df_clustered.shp")
        clusters_list = pd.DataFrame(
            index=geo_df_clustered.Cluster.unique(),
            columns=['Cluster', 'Population', 'Load [kW]'])

        clusters_list.loc[:, 'Cluster'] = geo_df_clustered.Cluster.unique()
        clusters_list = clusters_list.drop(index=-1)
        for i in clusters_list.Cluster:
            clusters_list.loc[i, 'Population'] = \
                round(sum(geo_df_clustered.loc
                          [geo_df_clustered['Cluster'] == i,
                           'Population']))
            clusters_list.loc[i, 'Load [kW]'] = \
                round(clusters_list.loc[i, 'Population'] * pop_load, 2)
        print("Clusters successfully imported")
        l()
        os.chdir(r'..//..')
        if step == 3:
            return df_weighted, input_sub, input_csv, crs, resolution, unit, \
                   pop_load, pop_thresh, line_bc, sub_cost_hv, sub_cost_mv, \
                   branch, pop_thresh_lr, line_bc_col, full_ele, wt, \
                   coe, grid_ir, grid_om, grid_lifetime, geo_df_clustered, \
                   clusters_list

        if step == 4:
            if branch == 'yes':
                grid_resume = pd.read_csv('Output/Branches/grid_resume_opt.csv'
                                          , index_col='Cluster')
            else:
                grid_resume = pd.read_csv('Output/Grids/grid_resume_opt.csv',
                                          index_col='Cluster')
            return geo_df_clustered, clusters_list, wt, grid_resume, \
                coe, grid_ir, grid_om, grid_lifetime

        if step == 5:
            if branch == 'yes':
                grid_resume = pd.read_csv('Output/Branches/grid_resume_opt.csv'
                                          , index_col='Cluster')
            else:
                grid_resume = pd.read_csv('Output/Grids/grid_resume_opt.csv',
                                          index_col='Cluster')
            mg = pd.read_csv('Output/Microgrids/microgrids.csv',
                             index_col='Cluster')
            total_energy = pd.read_csv('Output/Microgrids/Grid_energy.csv',
                                       index_col='Cluster')
            return clusters_list, grid_resume, mg, total_energy, \
                coe, grid_ir, grid_om, grid_lifetime


def weighting(df):
    """
    Assign weights to all points of a dataframe according to the terrain
    characteristics and the distance to the nearest road.
    :param df: From which part of GISEle the user is starting
    :return df_weighted: Point dataframe with weight attributes assigned
    """
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
        # Land cover using the column Other or GLC to compute the weight
        option = 'GLC'  # user choice for the land cover data weighting
        if option == 'GLC':
            df_weighted.loc[index, 'Weight'] += landcover_csv.WeightGLC[
                landcover_csv.iloc[:, 0] == row.Land_cover].values[0]
        if option == 'other':
            df_weighted.loc[index, 'Weight'] += landcover_csv.WeightOther[
                landcover_csv.iloc[:, 2] == row.Land_cover].values[0]
        # Road distance conditions
        if row.Road_dist < 100:
            df_weighted.loc[index, 'Weight'] = 1
        elif row.Road_dist < 1000:
            df_weighted.loc[index, 'Weight'] += 5 * row.Road_dist / 1000
        else:
            df_weighted.loc[index, 'Weight'] += 5

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
    """
    Based on the input weighted dataframe, creates a geodataframe assigning to
    it the Coordinate Reference System and a Point geometry
    :param df_weighted: Input Point dataframe with weight attributes assigned
    :param crs: Coordinate Reference System of the electrification project
    :param unit: Type of CRS unit used if degree or meters
    :param input_csv: Name of the input file for exporting the geodataframe
    :param step: From which part of GISEle the user is starting
    :return geo_df: Point geodataframe with weight, crs and geometry defined
    :return pop_points: Dataframe containing only the coordinates of the points
    """
    print("Creating the GeoDataFrame..")
    geometry = [Point(xy) for xy in zip(df_weighted['X'], df_weighted['Y'])]
    geo_df = gpd.GeoDataFrame(df_weighted, geometry=geometry,
                              crs=int(crs))

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

    loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
    pop_points = pd.DataFrame(data=loc).values

    return geo_df, pop_points
