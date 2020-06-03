import os
import sys
import glob
import warnings
import pandas as pd
import math
import geopandas as gpd
import networkx as nx
import numpy as np
from fiona.crs import from_epsg
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from shapely.geometry import Point, MultiPoint, box, LineString
from shapely.ops import nearest_points
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from Codes.Weight_matrix import *
from Codes.remove_cycles_weight_final import *

sys.path.insert(0, 'Codes')  # necessary for native GISEle Packages
# native GISEle Packages
from Codes.Steinerman import *
from Codes.Spiderman import *


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


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
    limit_HV = float(config[8, 1])
    limit_MV = float(config[9, 1])
    if step == 1:
        df = pd.read_csv(input_csv + '.csv', sep=',')
        print("Input files successfully imported.")
        os.chdir(r'..//')
        return df, input_sub, input_csv, crs, resolution, unit, pop_load, \
               pop_thresh, line_bc, limit_HV, limit_MV
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
                   pop_load, pop_thresh, line_bc, limit_HV, limit_MV, \
                   geo_df_clustered, clusters_list
        return df_weighted, input_sub, input_csv, crs, resolution, unit, \
               pop_load, pop_thresh, line_bc, limit_HV, limit_MV


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


def clustering_sensitivity(pop_points, geo_df):
    s()
    print("2.Clustering - Sensitivity Analysis")
    s()
    print(
        "In order to define which areas are sufficiently big and enough "
        "populated to be considered valuable for electrification,\ntwo "
        "parameters must be set: NEIGHBOURHOOD and MINIMUM POINTS.\n"
        "First, provide a RANGE for these parameters and also the number of "
        "SPANS between the range values.")

    check = "y"
    while check == "y":
        s()
        eps = input("Provide the limits for the NEIGHBOURHOOD parameter \n"
                    "(Example 250, 2500): ")
        eps = sorted(list(map(int, eps.split(','))))
        s()
        pts = input("Provide the limits for the MINIMUM POINTS parameter:\n"
                    "(Example 10, 100):  ")
        pts = sorted(list(map(int, pts.split(','))))
        s()
        spans = int(input("How many SPANS between these values?: "))
        s()
        span_eps = []
        span_pts = []
        for j in range(1, spans + 1):
            if spans != 1:
                eps_ = int(eps[0] + (eps[1] - eps[0]) / (spans - 1) * (j - 1))
            else:
                eps_ = int(eps[0] + (eps[1] - eps[0]) / spans * (j - 1))
            span_eps.append(eps_)
        for i in range(1, spans + 1):
            if spans != 1:
                pts_ = int(pts[0] + (pts[1] - pts[0]) / (spans - 1) * (i - 1))
            else:
                pts_ = int(pts[0] + (pts[1] - pts[0]) / spans * (i - 1))
            span_pts.append(pts_)
        tab_area = tab_people = tab_people_area = tab_cluster = \
            pd.DataFrame(index=span_eps, columns=span_pts)
        total_people = int(geo_df['Population'].sum(axis=0))
        for eps in span_eps:
            for pts in span_pts:
                db = DBSCAN(eps=eps, min_samples=pts, metric='euclidean'). \
                    fit(pop_points, sample_weight=geo_df['Population'])
                labels = db.labels_
                n_clusters_ = int(len(set(labels)) -
                                  (1 if -1 in labels else 0))
                geo_df['clusters'] = labels
                gdf_pop_noise = geo_df[geo_df['clusters'] == -1]
                noise_people = round(sum(gdf_pop_noise['Population']), 0)
                clustered_area = len(geo_df) - len(gdf_pop_noise)
                perc_area = int(clustered_area / len(geo_df) * 100)
                clustered_people = int((1 - noise_people / total_people) * 100)
                if clustered_area != 0:  # check to avoid crash by divide zero
                    people_area = (total_people - noise_people) \
                                  / clustered_area
                else:
                    people_area = 0
                tab_cluster.at[eps, pts] = n_clusters_
                tab_people.at[eps, pts] = clustered_people
                tab_area.at[eps, pts] = perc_area
                tab_people_area.at[eps, pts] = people_area
        print(
            "Number of clusters - columns MINIMUM POINTS - rows NEIGHBOURHOOD")
        print(tab_cluster)
        l()
        print(
            "% of clustered people - columns MINIMUM POINTS - rows "
            "NEIGHBOURHOOD")
        print(tab_people)
        l()
        print(
            "% of clustered area - columns MINIMUM POINTS - rows "
            "NEIGHBOURHOOD")
        print(tab_area)
        l()
        print("People per area - columns MINIMUM POINTS - rows NEIGHBOURHOOD")
        print(tab_people_area)
        l()
        check = str(input("Would you like to run another sensitivity "
                          "analysis with a more precise interval? (y/n): "))
        s()
    # os.chdir(r'Output//Clusters//Sensitivity')
    # tab_cluster.to_csv("n_clusters.csv")
    # tab_people.to_csv("%_peop.csv")
    # tab_area.to_csv("%_area.csv")
    # tab_people_area.to_csv("people_area.csv")
    # os.chdir(r'..//..//..')
    # print("Clustering sensitivity process completed.\n"
    #       "You can check the tables in the Output folder.")
    l()

    return


def clustering(pop_points, geo_df, pop_load):
    print('Choose the final combination of NEIGHBOURHOOD and MINIMUM POINTS')
    l()
    # eps = float(input("Chosen NEIGHBOURHOOD: "))
    eps = 1500
    s()
    pts = 500
    # pts = int(input("Chosen MINIMUM POINTS: "))
    s()
    db = DBSCAN(eps=eps, min_samples=pts, metric='euclidean').fit(
        pop_points, sample_weight=geo_df['Population'])
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # ignore noise
    clusters_list = np.arange(n_clusters)
    print('Initial number of clusters: %d' % n_clusters)
    # print('Estimated number of noise points: %d' % list(labels).count(-1))
    check = "y"

    while check == "y":
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            markersize = 3
            fontsize = 15
            if k == -1:  # Black used for noise.
                col = [0, 0, 0, 1]
                markersize = 1
                fontsize = 0
            cluster_points = (labels == k)
            xy = pop_points[cluster_points]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markeredgewidth=0.0,
                     markersize=markersize)
            for i, ((x, y, z),) in enumerate(zip(xy)):
                plt.text(x, y, str(k), ha="center", va="center",
                         fontsize=fontsize, color='black')
                if i == 0:
                    break
        plt.show()
        plt.clf()

        s()
        check = str(input('Would you like to merge clusters? (y/n): '))
        s()
        if check == "n":
            break
        print('The current clusters available for merging are:'
              '\n' + str(clusters_list))
        s()
        merge1 = int(input('Select a base cluster that you want to merge: '))
        merge2 = input('Which clusters would you like to merge into the '
                       'cluster' + str(merge1) + '? (Example: 1,13,25): ')
        merge2 = sorted(list(map(int, merge2.split(','))))
        s()
        for i in merge2:
            clusters_list = np.delete(clusters_list,
                                      np.where(clusters_list == i))
            labels[labels == i] = merge1
    geo_df_clustered = geo_df
    geo_df_clustered['clusters'] = labels
    clusters_list = np.tile(clusters_list, (3, 1))
    for i in clusters_list[0]:
        pop_i = sum(geo_df_clustered.loc
                    [geo_df_clustered['clusters'] == i, 'Population'])
        clusters_list[1, np.where(clusters_list[0] == i)] = pop_i
        clusters_list[2, np.where(clusters_list[0] == i)] = pop_i * pop_load
    os.chdir(r'Output//Clusters')
    geo_df_clustered.to_file("geo_df_clustered.shp")
    gdf_clustered_clean = geo_df_clustered[geo_df_clustered['clusters'] != -1]
    gdf_clustered_clean.to_file("geo_df_clustered_clean.shp")
    print("Clustering completed and files exported")
    os.chdir(r'..//..')
    l()
    return geo_df_clustered, clusters_list


def nearest(row, df, src_column=None):
    """Find the nearest point and return the value from specified column."""
    # Find the geometry that is closest
    nearest_p = df['geometry'] == nearest_points(row['geometry'],
                                                 df.unary_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df[nearest_p][src_column].values[0]
    return value


def distance_3d(df1, df2, x, y, z):
    """Find the 3D euclidean distance between two dataframes of points."""

    d1_coordinates = {'x': df1[x], 'y': df1[y], 'z': df1[z]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']

    d2_coordinates = {'x': df2[x], 'y': df2[y], 'z': df2[z]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = pd.DataFrame(cdist(df1_loc.values, df2_loc.values, 'euclidean'),
                         index=df1_loc.index, columns=df2_loc.index)
    return value


def substation_assignment(cluster_n, geo_df, gdf_cluster_pop, substations,
                          clusters_list, limit_HV, limit_MV, crs):

    if clusters_list[2][np.where(clusters_list[0] == cluster_n)[0][0]] \
            > limit_HV:
        substations = substations[substations['Type'] == 'HV']
    elif clusters_list[2][
        np.where(clusters_list[0] == cluster_n)[0][0]] > limit_MV and \
            clusters_list[2][np.where(clusters_list[0] == cluster_n)[0][0]] \
            < limit_HV:
        substations = substations[substations['Type'] == 'MV']
    elif clusters_list[2][
        np.where(clusters_list[0] == cluster_n)[0][0]] < limit_MV:
        substations = substations[substations['Type'] != 'HV']

    substations['nearest_id'] = substations.apply(nearest, df=geo_df,
                                                  src_column='ID', axis=1)

    sub_in_df = gpd.GeoDataFrame(crs=from_epsg(crs))
    for i, row in substations.iterrows():
        sub_in_df = sub_in_df.append(
            geo_df[geo_df['ID'] == row['nearest_id']], sort=False)
    sub_in_df.reset_index(drop=True, inplace=True)

    dist = distance_3d(gdf_cluster_pop, sub_in_df, 'X', 'Y', 'Elevation')

    assigned_substation = sub_in_df[sub_in_df['ID'] == dist.min().idxmin()]
    connection_type = substations[substations['nearest_id']
                                  == dist.min().idxmin()].Type.values[0]
    l()
    return assigned_substation, connection_type


def cluster_grid_routing(geo_df, gdf_cluster_pop, crs, resolution, line_bc,
                         points_to_electrify):

    if points_to_electrify < 3:
        c_grid = gdf_cluster_pop
        c_grid_cost = 0
        c_grid_length = 0

    elif points_to_electrify < 1000:
        c_grid1, c_grid_cost1, c_grid_length1, c_grid_points1 = \
            Steinerman(geo_df, gdf_cluster_pop, crs, line_bc, resolution)
        c_grid2, c_grid_cost2, c_grid_length2, c_grid_points2 = \
            Spiderman(geo_df, gdf_cluster_pop, crs, line_bc, resolution)

        if c_grid_cost1 <= c_grid_cost2:
            print(" Steiner Wins!")
            c_grid = c_grid1
            c_grid_length = c_grid_length1
            c_grid_cost = c_grid_cost1
            c_grid_points = c_grid_points1

        elif c_grid_cost1 > c_grid_cost2:
            print(" Steiner Loses!")
            c_grid = c_grid2
            c_grid_length = c_grid_length2
            c_grid_cost = c_grid_cost2
            c_grid_points = c_grid_points2

    elif points_to_electrify >= 1000:
        print("Too many points to use Steiner")
        c_grid, c_grid_cost, c_grid_length, c_grid_points = \
            Spiderman(geo_df, gdf_cluster_pop, crs, line_bc, resolution)

    return c_grid, c_grid_cost, c_grid_length, c_grid_points


def substation_connection(c_grid, assigned_substation, c_grid_points, geo_df,
                          line_bc, resolution, crs):

    checkpoint = list(c_grid['ID1'].values) + list(
        c_grid['ID2'].values)
    checkpoint = list(dict.fromkeys(checkpoint))
    checkpoint_ext = gpd.GeoDataFrame(crs=from_epsg(crs))
    for i in checkpoint:
        checkpoint_ext = checkpoint_ext.append(
            geo_df[geo_df['ID'] == i], sort=False)
    checkpoint_ext.reset_index(drop=True, inplace=True)
    assigned_substation[
        'nearest_gridpoint'] = assigned_substation.apply(nearest,
                                                         df=checkpoint_ext,
                                                         src_column='ID',
                                                         axis=1)
    # -------------------------
    Pointsub = Point(assigned_substation['X'],
                     assigned_substation['Y'])
    idgrid = int(assigned_substation['nearest_gridpoint'].values)
    Pointgrid = Point(float(geo_df[geo_df['ID'] == idgrid].X),
                      float(geo_df[geo_df['ID'] == idgrid].Y))
    dist = Pointsub.distance(Pointgrid)
    p1 = geo_df[geo_df['ID'] == idgrid]
    if dist < 30000:
        connection, connection_cost, connection_length = \
            grid_direct_connection(geo_df, assigned_substation, p1, crs,
                                   line_bc, resolution)
    else:
        print('Cluster too far from the main grid to be connected, a'
              ' microgrid solution is suggested')
        connection_cost = 0
        connection_length = 99999

    return connection, connection_cost, connection_length


def grid(geo_df_clustered, geo_df, crs, clusters_list, resolution,
         pop_threshold, input_sub, line_bc, limit_HV, limit_MV):
    s()
    print("3. Grid Creation")
    s()

    grid_merged = connection_merged = pd.DataFrame()
    grid_resume = pd.DataFrame(index=clusters_list[0],
                               columns=['Cluster', 'Grid_Length', 'Grid_Cost',
                                        'Connection_Length',
                                        'Connection_Cost', 'ClusterLoad',
                                        'ConnectionType'])
    grid_resume['Cluster'] = clusters_list[0]
    grid_resume['ClusterLoad'] = clusters_list[2]
    os.chdir(r'Input//')
    substations = gpd.read_file(input_sub + '.shp', sep=',')
    substations['ID'] = range(0, len(substations))
    os.chdir(r'..')
    os.chdir(r'Output//Grids')
    for cluster_n in clusters_list[0]:

        gdf_cluster = geo_df_clustered[
            geo_df_clustered['clusters'] == cluster_n]
        gdf_cluster_pop = gdf_cluster[
            gdf_cluster['Population'] >= pop_threshold]

        points_to_electrify = int(len(gdf_cluster_pop))

        if points_to_electrify > 0:

            assigned_substation, connection_type = \
                substation_assignment(cluster_n, geo_df, gdf_cluster_pop,
                                      substations, clusters_list, limit_HV,
                                      limit_MV, crs)

            grid_resume.loc[cluster_n, 'ConnectionType'] = connection_type
            l()
            print("Creating grid for Cluster n." + str(cluster_n) + " of "
                  + str(len(clusters_list[0])))
            l()
            c_grid, c_grid_cost, c_grid_length, c_grid_points = \
                cluster_grid_routing(geo_df, gdf_cluster_pop, crs, resolution,
                                     line_bc, points_to_electrify)

            connection, connection_cost, connection_length = \
                substation_connection(c_grid, assigned_substation, c_grid_points,
                                      geo_df, line_bc, resolution, crs)

            fileout = 'Grid_' + str(cluster_n) + '.shp'
            c_grid.to_file(fileout)
            grid_merged = gpd.GeoDataFrame(
                pd.concat([grid_merged, c_grid], sort=True))
            fileout = 'connection_' + str(cluster_n) + '.shp'

            if connection.type.values[0] == 'LineString':
                connection.to_file(fileout)
                connection_merged = gpd.GeoDataFrame(pd.concat(
                    [connection_merged, connection], sort=True))

            grid_resume.at[cluster_n, 'Grid_Length'] = c_grid_length / 1000
            grid_resume.at[cluster_n, 'Grid_Cost'] = c_grid_cost / 1000
            grid_resume.at[cluster_n, 'Connection_Length'] = connection_length / 1000
            grid_resume.at[cluster_n, 'Connection_Cost'] = connection_cost / 1000

        elif points_to_electrify == 0:

            grid_resume.at[cluster_n, 'Grid_Length'] = 0
            grid_resume.at[cluster_n, 'Grid_Cost'] = 0
            grid_resume.at[cluster_n, 'Connection_Length'] = 0
            grid_resume.at[cluster_n, 'Connection_Cost'] = 0

    grid_resume.to_csv('grid_resume.csv')
    connection_merged.crs = grid_merged.crs = geo_df.crs
    connection_merged.to_file('merged_connections')
    grid_merged.to_file('merged_grid')
    os.chdir(r'..//..')
    print("All grids have been optimized and exported.")
    l()
    return grid_resume, line_bc


def load():
    print("8.Load creation")
    s()
    peop_per_house = int(
        input("How many people are there on average in an household?: "))
    s()

    os.chdir(r'Output//Clusters')
    clusters = gpd.read_file('total_cluster.shp')
    n_clusters = max(clusters['clusters']) + 1
    os.chdir(r'..//..')

    os.chdir(r'Input//3_load_ref')
    clusters_list_2 = list(set(clusters['clusters']))

    load_ref = pd.read_csv('Load.csv', index_col=0)
    users_ref = load_ref.loc[['Numero di utenti per categoria']]
    house_ref = int(
        load_ref['Case "premium"']['Numero di utenti per categoria']
        + load_ref['Case "standard"']['Numero di utenti per categoria'])

    hours = load_ref.index.values
    hours = np.delete(hours, 0, 0)

    os.chdir(r'..//..//Output//Load')
    total_loads = pd.DataFrame(index=hours)

    loads_list = []
    houses_list = []
    for n in clusters_list_2:
        cluster = clusters[clusters['clusters'] == n]
        cluster_pop = sum(cluster['Population'])

        cluster_houses = int(cluster_pop / peop_per_house)

        load_cluster = load_ref.copy()

        public_services = ["Clinica", "Scuola Primaria", "Scuola Secondaria",
                           "Amministrazione"]

        for service in load_cluster.columns.values:

            users_ref = load_ref.loc[['Numero di utenti per categoria']]

            if service in public_services:
                load_cluster.at['Numero di utenti per categoria', service] = 1

            else:

                new_n_users = round(
                    users_ref[service][
                        'Numero di utenti per categoria'] / house_ref * cluster_houses + 0.5,
                    0)
                load_cluster.at[
                    'Numero di utenti per categoria', service] = new_n_users

        for i in hours:
            for j in load_cluster.columns.values:
                new_load = round(load_ref[j][i] / house_ref * cluster_houses,
                                 3)
                load_cluster.at[i, j] = new_load

                users_ref = load_cluster.loc[
                    ['Numero di utenti per categoria']]  # NON CANCELLARE

        name_load = "load_cluster_n_" + str(n) + '_houses_' + str(
            cluster_houses) + '.csv'
        load_cluster.to_csv(name_load)
        loads_list.append(name_load)
        houses_list.append(cluster_houses)
        total_loads[n] = load_cluster[load_cluster.columns].sum(axis=1)

    total_loads.to_csv("total_loads.csv")
    os.chdir(r'..//..')

    print("All loads successfully computed")
    s()

    print("Total load - columns CLUSTERS - rows HOURS")
    print(total_loads)
    l()
    return loads_list, houses_list, house_ref


def grid_direct_connection(mesh, gdf_cluster_pop, substation_designata,
                           Proj_coords, paycheck, resolution):
    print(
        'Only few points to electrify, no internal cluster grid necessary. Connecting to the nearest substation')
    Pointsub = Point(substation_designata['X'], substation_designata['Y'])
    idsub = int(substation_designata['ID'].values)
    if len(gdf_cluster_pop.ID) == 1:
        Pointgrid = Point(float(gdf_cluster_pop['X']),
                          float(gdf_cluster_pop['Y']))
        idgrid = int(gdf_cluster_pop['ID'].values)
        dist = Pointsub.distance(Pointgrid)
    elif len(
            gdf_cluster_pop.ID) > 0:  # checks if there's more than 1 point, connected the closest one
        d_point = {}
        d_dist = {}
        for i in range(0, len(gdf_cluster_pop.ID)):
            d_point[i] = Point(float(gdf_cluster_pop.X.values[i]),
                               float(gdf_cluster_pop.Y.values[i]))
            d_dist[i] = Pointsub.distance(d_point[i])
        if d_dist[1] > d_dist[0]:
            Pointgrid = d_point[0]
            dist = d_dist[0]
            idgrid = int(gdf_cluster_pop.ID.values[0])
        else:
            Pointgrid = d_point[1]
            dist = d_dist[1]
            idgrid = int(gdf_cluster_pop.ID.values[1])
    if dist < 1000:
        extension = dist
    elif dist < 2000:
        extension = dist / 2
    else:
        extension = dist / 4

    xmin = min(Pointsub.x, Pointgrid.x)
    xmax = max(Pointsub.x, Pointgrid.x)
    ymin = min(Pointsub.y, Pointgrid.y)
    ymax = max(Pointsub.y, Pointgrid.y)
    bubble = box(minx=xmin - extension, maxx=xmax + extension,
                 miny=ymin - extension, maxy=ymax + extension)
    df_box = mesh[mesh.within(bubble)]
    df_box.index = pd.Series(range(0, len(df_box['ID'])))

    solocoord2 = {'x': df_box['X'], 'y': df_box['Y']}
    solocoord3 = {'x': df_box['X'], 'y': df_box['Y'], 'z': df_box['Elevation']}
    Battlefield2 = pd.DataFrame(data=solocoord2)
    Battlefield2.index = df_box['ID']
    Battlefield3 = pd.DataFrame(data=solocoord3)
    Battlefield3.index = df_box['ID']
    print('Checkpoint 2')

    Distance_2D_box = distance_matrix(Battlefield2, Battlefield2)
    Distance_3D_box = pd.DataFrame(
        cdist(Battlefield3.values, Battlefield3.values, 'euclidean'),
        index=Battlefield3.index, columns=Battlefield3.index)
    Weight_matrix = weight_matrix(df_box, Distance_3D_box, paycheck)
    print('3D distance matrix has been created')
    min_length = Distance_2D_box[Distance_2D_box > 0].min()
    diag_length = resolution * 1.5

    # Definition of weighted connections matrix
    Edges_matrix = Weight_matrix
    # Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 999999999
    Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 0
    Edges_matrix_sparse = sparse.csr_matrix(Edges_matrix)
    print('Checkpoint 4: Edges matrix completed')
    # Graph = nx.from_numpy_matrix(Edges_matrix.to_numpy())
    Graph = nx.from_scipy_sparse_matrix(Edges_matrix_sparse)
    #    Graph.remove_edges_from(np.where(Edges_matrix.to_numpy() == 999999999))
    source = df_box.loc[df_box['ID'].values == idsub, :]
    target = df_box.loc[df_box['ID'].values == idgrid, :]
    source = source.index[0]
    target = target.index[0]
    print('Checkpoint 5')
    # Lancio Dijkstra e ottengo il percorso ottimale per connettere 'source' a 'target'
    path = nx.dijkstra_path(Graph, source, target, weight='weight')
    Steps = len(path)
    # Analisi del path
    r = 0  # Giusto un contatore
    PathID = []
    Digievoluzione = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
    Digievoluzione['id'] = pd.Series(range(1, Steps))
    Digievoluzione['x1'] = pd.Series(range(1, Steps))
    Digievoluzione['x2'] = pd.Series(range(1, Steps))
    Digievoluzione['y1'] = pd.Series(range(1, Steps))
    Digievoluzione['y2'] = pd.Series(range(1, Steps))
    Digievoluzione['ID1'] = pd.Series(range(1, Steps))
    Digievoluzione['ID2'] = pd.Series(range(1, Steps))
    Digievoluzione['Weight'] = pd.Series(range(1, Steps))
    Digievoluzione['geometry'] = None
    Digievoluzione['geometry'].stype = gpd.geoseries.GeoSeries
    print('Checkpoint 6:')

    for h in range(0, Steps - 1):
        con = [min(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']),
               max(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID'])]
        # if con not in connections:
        PathID.append(df_box.loc[path[h], 'ID'])
        PathID.append(df_box.loc[path[h + 1], 'ID'])
        Digievoluzione.at[r, 'geometry'] = LineString(
            [(df_box.loc[path[h], 'X'], df_box.loc[path[h], 'Y'],
              df_box.loc[path[h], 'Elevation']),
             (
                 df_box.loc[path[h + 1], 'X'], df_box.loc[path[h + 1], 'Y'],
                 df_box.loc[path[h + 1], 'Elevation'])])
        Digievoluzione.at[r, ['id']] = 0
        Digievoluzione.at[r, ['x1']] = df_box.loc[path[h], 'X']
        Digievoluzione.at[r, ['x2']] = df_box.loc[path[h + 1], 'X']
        Digievoluzione.at[r, ['y1']] = df_box.loc[path[h], ['Y']]
        Digievoluzione.at[r, ['y2']] = df_box.loc[path[h + 1], ['Y']]
        Digievoluzione.at[r, 'ID1'] = df_box.loc[path[h], 'ID']
        Digievoluzione.at[r, 'ID2'] = df_box.loc[path[h + 1], 'ID']
        Digievoluzione.at[r, 'Weight'] = Edges_matrix.loc[
            df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']]
        # connections.append(con)
        r += 1

    print('Checkpoint 7: Genomica di digievoluzione identificata')
    # Elimino le righe inutili di Digievoluzione e i doppioni da PathID
    indexNames = Digievoluzione[Digievoluzione['id'] > 0].index
    Digievoluzione.drop(indexNames, inplace=True)
    PathID = list(dict.fromkeys(PathID))
    cordone_ombelicale = Digievoluzione
    cordone_ombelicale = cordone_ombelicale.drop(
        ['id', 'x1', 'x2', 'y1', 'y2'], axis=1)
    cordone_ombelicale['line_length'] = cordone_ombelicale['geometry'].length
    total_cost = int(cordone_ombelicale['Weight'].sum(axis=0))
    total_length = cordone_ombelicale['line_length'].sum(axis=0)

    return cordone_ombelicale, total_cost, total_length


def sizing(loads_list, clusters_list_2, houses_list, house_ref):
    print("9.Generation dimensioning")
    print(
        "Now you need to use an external tool in order to optimize the power generation block for each cluster.\n"
        "The energy consumption for each of them is in a CSV file named Output/Load/total_loads.csv")
    print(
        "From our side we'll use the Calliope Energy Model in order to optimize the system. This will need you to\n"
        "fill the model with details about each technology. The CSV files are in GISEle\Input\_5_components.\n"
        "At the moment, standard details are already in.")

    'Asking the user info about the project'
    # project_life = int(input("What's the project lifefime?: [years] ")) + 1
    project_life = 25
    time_span = range(project_life)

    os.chdir(r'Output//Clusters')
    clusters = gpd.read_file('total_cluster.shp')
    n_clusters = max(clusters['clusters']) + 1

    clusters_list = np.arange(n_clusters)
    # clusters_list_2 = clusters_list[:]
    clusters_list_2 = list(set(clusters['clusters']))
    os.chdir(r'..//..')

    total_energy = pd.DataFrame(index=clusters_list_2,
                                columns=['Total Energy Consumption [kWh]'])
    mg_NPC = pd.DataFrame(index=clusters_list_2, columns=['Total Cost [$]'])

    i = 0
    'Creating the final load profiles'
    for clusters in loads_list:
        c = clusters_list_2[i]
        n = houses_list[i]
        os.chdir(r'Input//7_sizing_calliope//timeseries_data')

        load_ref = pd.read_csv('demand_power_ref.csv', index_col=0, header=0)
        load_cluster = load_ref.copy()

        load_cluster['X1'] = load_ref['X1'] / house_ref * n
        energy = -1 * load_cluster.values.sum()
        total_energy.at[
            c, 'Total Energy Consumption [kWh]'] = energy * project_life

        load_final = 'whole_' + clusters + '.csv'

        load_cluster.to_csv(load_final)
        load_cluster.to_csv('demand_power.csv')
        os.chdir(r'..//..//..')

        lat = -16.5941
        long = 36.5955

        '''import geopandas as gpd
        import pandas as pd
        from pyproj import transform, Proj
        from fiona.crs import from_epsg
        from shapely.geometry import Point, MultiPoint

        point = gpd.GeoDataFrame(crs=from_epsg(32737))
        point.loc[0, 'x'] = 227911
        point.loc[0, 'y'] = 8097539
        point['geometry'] = None
        point['geometry'].stype = gpd.geoseries.GeoSeries
        point.loc[0, 'geometry'] = Point(point.loc[0, 'x'], point.loc[0, 'y'])

        point['geometry'] = point['geometry'].to_crs(epsg=4326)

        point.loc[0, 'x_deg'] = point.loc[0, 'geometry'].x
        point.loc[0, 'y_deg'] = point.loc[0, 'geometry'].y

        print(point)'''

        pre_calliope(i, project_life, lat, long)

        cost_km_line = 20000
        total_undiscounted_cost = calliope_x_GISEle(time_span, c, cost_km_line,
                                                    project_life)
        l()
        i += 1
        os.chdir(r'..//..')

        mg_NPC.at[c, 'Total Cost [$]'] = total_undiscounted_cost

    return total_energy, mg_NPC


def final_results(clusters_list_2, total_energy, grid_resume, mg_NPC):
    os.chdir(r'Output//Sizing_calliope')
    grid_resume = pd.read_csv('grid_resume.csv', header=0, index_col=0)

    os.chdir(r'..//Clusters')
    clusters = gpd.read_file('total_cluster.shp')
    n_clusters = max(clusters['clusters']) + 1

    clusters_list = np.arange(n_clusters)
    # clusters_list_2 = clusters_list[:]
    clusters_list_2 = list(set(clusters['clusters']))

    os.chdir(r'..//..')

    final_LCOEs = pd.DataFrame(index=clusters_list_2,
                               columns=['GRID_NPC [$]', 'MG_NPC [$]',
                                        'Total Energy Consumption [kWh]',
                                        'MG_LCOE [$/kWh]', 'GRID_LCOE [$/kWh]',
                                        'Suggested Solution'])

    print(total_energy)
    s()
    print(grid_resume)
    s()
    print(mg_NPC)
    s()

    for clu in clusters_list_2:
        print(clu)
        final_LCOEs.at[clu, 'Total Energy Consumption [kWh]'] = \
            total_energy.loc[clu, 'Total Energy Consumption [kWh]']
        final_LCOEs.at[clu, 'GRID_NPC [$]'] = grid_resume.loc[
                                                  clu, 'Grid_Cost'] + \
                                              grid_resume.loc[
                                                  clu, 'Connection_Cost']

        final_LCOEs.at[clu, 'MG_NPC [$]'] = mg_NPC.loc[clu, 'Total Cost [$]']

        mg_lcoe = final_LCOEs.loc[clu, 'MG_NPC [$]'] / final_LCOEs.loc[
            clu, 'Total Energy Consumption [kWh]']

        final_LCOEs.at[clu, 'MG_LCOE [$/kWh]'] = mg_lcoe

        grid_lcoe = final_LCOEs.loc[clu, 'GRID_NPC [$]'] / final_LCOEs.loc[
            clu, 'Total Energy Consumption [kWh]'] + 0.1428

        final_LCOEs.at[clu, 'GRID_LCOE [$/kWh]'] = grid_lcoe

        if mg_lcoe > grid_lcoe:
            ratio = grid_lcoe / mg_lcoe
            if ratio < 0.9:
                proposal = 'GRID'
            else:
                proposal = 'Although the GRID option is cheaper, ' \
                           'the two costs are very similar, so further investigation is suggested'

        else:
            ratio = mg_lcoe / grid_lcoe
            if ratio < 0.9:
                proposal = 'OFF-GRID'
            else:
                proposal = 'Although the OFF-GRID option is cheaper, ' \
                           'the two costs are very similar, so further investigation is suggested'

        final_LCOEs.at[clu, 'Suggested Solution'] = proposal

    l()
    print(final_LCOEs)
    l()

    os.chdir(r'Output//Final_results')
    final_LCOEs.to_csv('final_LCOE.csv')

    return final_LCOEs


def grid_optimization(gdf_clusters, geodf_in, grid_resume, proj_coords,
                      resolution, paycheck):
    os.chdir(r'Output//Grids')
    new_connection_merged = pd.DataFrame()
    grid_resume = pd.read_csv('grid_resume.csv')
    grid_resume.reset_index(drop=True, inplace=True)
    grid_resume = grid_resume.sort_values(by='Connection_Cost')
    clusters_list_2 = grid_resume['Cluster']
    check = np.zeros(len(clusters_list_2), dtype=bool)
    grid_resume = grid_resume.sort_values(by='Cluster')
    # check = np.zeros(len(clusters_list_2), dtype=bool)

    count = 0

    while not any(check):  # checks if all variables in check are True
        for i in clusters_list_2:
            betterconnection = False
            print(
                'Evaluating if there is a better connection for cluster ' + str(
                    i))
            k = np.where(grid_resume[
                             'Cluster'] == i)  # find the index of the cluster we are analysing
            dist = pd.DataFrame(index=clusters_list_2,
                                columns=['Distance', 'NearestPoint', 'Cost'])
            for j in clusters_list_2:
                k2 = np.where(grid_resume[
                                  'Cluster'] == j)  # index of the cluster we trying to connect
                if grid_resume.Connection_Length.values[
                    k] == 0:  # k is necessary due to cluster merging causing indexes and cluster numbers to not match
                    check[count] = True
                    break
                if i == j:
                    dist.Distance[j] = 9999999
                    dist.Cost[j] = 99999999
                    continue

                grid1 = gpd.read_file("Grid_" + str(j) + ".shp")
                chunk_points = pd.DataFrame()
                for id in grid1.ID1:  # getting all the points inside the chunk to compute the nearest_points
                    chunk_points = gpd.GeoDataFrame(
                        pd.concat([chunk_points,
                                   gdf_clusters[gdf_clusters['ID'] == id]],
                                  sort=True))
                last_id = grid1.loc[
                    grid1.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
                chunk_points = gpd.GeoDataFrame(
                    pd.concat([chunk_points,
                               gdf_clusters[gdf_clusters['ID'] == last_id]],
                              sort=True))
                uu_grid1 = chunk_points.unary_union

                grid2 = gpd.read_file("Grid_" + str(i) + ".shp")
                chunk_points = pd.DataFrame()
                for id in grid2.ID1:  # getting all the points inside the chunk to compute the nearest_points
                    chunk_points = gpd.GeoDataFrame(
                        pd.concat([chunk_points,
                                   gdf_clusters[gdf_clusters['ID'] == id]],
                                  sort=True))
                last_id = grid2.loc[
                    grid2.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
                chunk_points = gpd.GeoDataFrame(
                    pd.concat([chunk_points,
                               gdf_clusters[gdf_clusters['ID'] == last_id]],
                              sort=True))
                uu_grid2 = chunk_points.unary_union
                del chunk_points

                dist.NearestPoint[j] = nearest_points(uu_grid1, uu_grid2)
                p1 = gdf_clusters[
                    gdf_clusters['geometry'] == dist.NearestPoint[j][0]]
                p2 = gdf_clusters[
                    gdf_clusters['geometry'] == dist.NearestPoint[j][1]]
                # if the distance between grids is higher than 1.2 times the length  former connection, skip
                if (dist.NearestPoint[j][0].distance(
                        dist.NearestPoint[j][1])) / 1000 > \
                        1.2 * (grid_resume.at[k[0][0], 'Connection_Length']):
                    dist.Distance[j] = 9999999
                    dist.Cost[j] = 99999999
                    continue
                block_print()
                direct_connection, direct_cost, direct_length = grid_direct_connection(
                    geodf_in, p1, p2,
                    proj_coords,
                    paycheck,
                    resolution)
                enable_print()

                dist.Distance[j] = direct_length
                if grid_resume.ConnectionType.values[
                    np.where(grid_resume['Cluster'] == j)][0] == 'HV':
                    direct_cost = direct_cost
                dist.Cost[j] = direct_cost
                if grid_resume.ConnectionType.values[
                    np.where(grid_resume['Cluster'] == j)][0] == 'MV':
                    if grid_resume.ClusterLoad.at[k[0][0]] + \
                            grid_resume.ClusterLoad.at[k2[0][0]] > 3000:
                        continue
                if grid_resume.ConnectionType.values[
                    np.where(grid_resume['Cluster'] == j)][0] == 'MV':
                    if grid_resume.ClusterLoad.at[k[0][0]] + \
                            grid_resume.ClusterLoad.at[k2[0][0]] > 300:
                        continue
                if min(dist.Cost) == direct_cost:
                    if direct_cost / 1000 < grid_resume.Connection_Cost.values[
                        k]:
                        if check[np.where(clusters_list_2 == j)]:
                            direct = 'NewConnection_' + str(i) + '.shp'
                            direct_connection.to_file(direct)
                            print('A new connection for Cluster ' + str(
                                i) + ' was successfully created')
                            grid_resume.at[k[0][
                                               0], 'Connection_Length'] = direct_length / 1000
                            grid_resume.at[k[0][
                                               0], 'Connection_Cost'] = direct_cost / 1000
                            grid_resume.at[k[0][0], 'ConnectionType'] = \
                                grid_resume.ConnectionType.values[
                                    np.where(grid_resume['Cluster'] == j)][0]
                            betterconnection = True
                            best_connection = direct_connection
            check[count] = True
            count = count + 1
            if betterconnection:
                new_connection_merged = gpd.GeoDataFrame(
                    pd.concat([new_connection_merged, best_connection],
                              sort=True))
            elif not betterconnection:
                if grid_resume.Connection_Length.values[k] > 0:
                    best_connection = gpd.read_file(
                        'connection_' + str(i) + '.shp')
                    new_connection_merged = gpd.GeoDataFrame(
                        pd.concat([new_connection_merged, best_connection],
                                  sort=True))

    grid_resume.to_csv('grid_resume_opt.csv')
    new_connection_merged.crs = geodf_in.crs
    new_connection_merged.to_file('total_connections_opt')
    return grid_resume
