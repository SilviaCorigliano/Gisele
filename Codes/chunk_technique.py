"""
Main branches and collateral algorithm

Created on Mon Jan 27 2020

@author: Vinicius Gadelha
"""

import os
import geopandas as gpd
import pandas as pd
import math
import numpy as np
import sys
from shapely.geometry import MultiPoint
from shapely.ops import nearest_points
from Codes.remove_cycles_weight_final import *
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import Point, LineString, box
from fiona.crs import from_epsg
import matplotlib.pyplot as plt
import networkx as nx
import time
from Codes.Weight_matrix import *
from Codes.Steiner_tree_code import *
from Codes.Steinerman import *
from Codes.Spiderman import *
import warnings

warnings.filterwarnings("ignore")


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    """Find the nearest point and return the corresponding value from specified column."""
    # Find the geometry that is closest
    nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].get_values()[0]
    return value


def grid_direct_connection(mesh, gdf_cluster_pop, substation_designata, Proj_coords, paycheck, resolution):
    Point1 = Point(float(substation_designata['X']), float(substation_designata['Y']))
    id1 = int(substation_designata['ID'].values)
    Point2 = Point(float(gdf_cluster_pop['X']), float(gdf_cluster_pop['Y']))
    id2 = int(gdf_cluster_pop['ID'].values)
    dist = Point1.distance(Point2)
    if dist < 1000:
        extension = dist
    elif dist < 2000:
        extension = dist / 2
    else:
        extension = dist / 4

    xmin = min(Point1.x, Point2.x)
    xmax = max(Point1.x, Point2.x)
    ymin = min(Point1.y, Point2.y)
    ymax = max(Point1.y, Point2.y)
    bubble = box(minx=xmin - extension, maxx=xmax + extension, miny=ymin - extension, maxy=ymax + extension)
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
    Distance_3D_box = pd.DataFrame(cdist(Battlefield3.values, Battlefield3.values, 'euclidean'),
                                   index=Battlefield3.index, columns=Battlefield3.index)
    Weight_matrix = weight_matrix(df_box, Distance_3D_box, paycheck)
    print('3D distance matrix has been created')
    # min_length = Distance_2D_box[Distance_2D_box > 0].min()
    diag_length = resolution * 1.7

    # Definition of weighted connections matrix
    Edges_matrix = Weight_matrix
    # Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 999999999
    Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 0
    Edges_matrix_sparse = sparse.csr_matrix(Edges_matrix)
    print('Checkpoint 4: Edges matrix completed')
    # Graph = nx.from_numpy_matrix(Edges_matrix.to_numpy())
    Graph = nx.from_scipy_sparse_matrix(Edges_matrix_sparse)
    #    Graph.remove_edges_from(np.where(Edges_matrix.to_numpy() == 999999999))
    source = df_box.loc[df_box['ID'].values == id1, :]
    target = df_box.loc[df_box['ID'].values == id2, :]
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
            [(df_box.loc[path[h], 'X'], df_box.loc[path[h], 'Y'], df_box.loc[path[h], 'Elevation']),
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
    cordone_ombelicale = cordone_ombelicale.drop(['id', 'x1', 'x2', 'y1', 'y2'], axis=1)
    cordone_ombelicale['line_length'] = cordone_ombelicale['geometry'].length
    total_cost = int(cordone_ombelicale['Weight'].sum(axis=0))
    total_length = cordone_ombelicale['line_length'].sum(axis=0)

    return cordone_ombelicale, total_cost, total_length


def grid_optimization(gdf_clusters, geodf_in, grid_resume, proj_coords, resolution, paycheck):
    os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Output\Chunks')
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
            print('Evaluating if there is a better connection for cluster ' + str(i))
            k = np.where(grid_resume['Cluster'] == i)  # find the index of the cluster we are analysing
            dist = pd.DataFrame(index=clusters_list_2, columns=['Distance', 'NearestPoint', 'Cost'])
            for j in clusters_list_2:
                k2 = np.where(grid_resume['Cluster'] == j)  # index of the cluster we trying to connect
                if grid_resume.Connection_Length.values[
                    k] == 0:  # k is necessary due to cluster merging causing indexes and cluster numbers to not match
                    check[count] = True
                    break
                if i == j:
                    dist.Distance[j] = 9999999
                    dist.Cost[j] = 99999999
                    continue

                grid1 = gpd.read_file("Chunk_" + str(j) + ".shp")
                chunk_points = pd.DataFrame()
                for id in grid1.ID1:  # getting all the points inside the chunk to compute the nearest_points
                    chunk_points = gpd.GeoDataFrame(
                        pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == id]], sort=True))
                last_id = grid1.loc[
                    grid1.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
                chunk_points = gpd.GeoDataFrame(
                    pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == last_id]], sort=True))
                uu_grid1 = chunk_points.unary_union

                grid2 = gpd.read_file("Chunk_" + str(i) + ".shp")
                chunk_points = pd.DataFrame()
                for id in grid2.ID1:  # getting all the points inside the chunk to compute the nearest_points
                    chunk_points = gpd.GeoDataFrame(
                        pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == id]], sort=True))
                last_id = grid2.loc[
                    grid2.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
                chunk_points = gpd.GeoDataFrame(
                    pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == last_id]], sort=True))
                uu_grid2 = chunk_points.unary_union
                del chunk_points

                dist.NearestPoint[j] = nearest_points(uu_grid1, uu_grid2)
                p1 = gdf_clusters[gdf_clusters['geometry'] == dist.NearestPoint[j][0]]
                p2 = gdf_clusters[gdf_clusters['geometry'] == dist.NearestPoint[j][1]]
                # if the distance between grids is higher than 1.2 times the length  former connection, skip
                if (dist.NearestPoint[j][0].distance(dist.NearestPoint[j][1])) / 1000 > \
                        1.2 * (grid_resume.at[k[0][0], 'Connection_Length']):
                    dist.Distance[j] = 9999999
                    dist.Cost[j] = 99999999
                    continue
                blockPrint()
                direct_connection, direct_cost, direct_length = grid_direct_connection(geodf_in, p1, p2,
                                                                                       proj_coords,
                                                                                       paycheck,
                                                                                       resolution)
                enablePrint()

                dist.Distance[j] = direct_length
                if grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0] == 'HV':
                    direct_cost = direct_cost + 1000000
                dist.Cost[j] = direct_cost
                if grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0] == 'MV_3P':
                    if grid_resume.ClusterLoad.at[k[0][0]] + grid_resume.ClusterLoad.at[k2[0][0]] > 3000:
                        continue
                if grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0] == 'MV_1P':
                    if grid_resume.ClusterLoad.at[k[0][0]] + grid_resume.ClusterLoad.at[k2[0][0]] > 300:
                        continue
                if min(dist.Cost) == direct_cost:
                    if direct_cost / 1000 < grid_resume.Connection_Cost.values[k]:
                        if check[np.where(clusters_list_2 == j)]:
                            direct = 'NewConnection_' + str(i) + '.shp'
                            direct_connection.to_file(direct)
                            print('A new connection for Cluster ' + str(i) + ' was successfully created')
                            grid_resume.at[k[0][0], 'Connection_Length'] = direct_length / 1000
                            grid_resume.at[k[0][0], 'Connection_Cost'] = direct_cost / 1000
                            grid_resume.at[k[0][0], 'ConnectionType'] = \
                                grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0]
                            betterconnection = True
                            best_connection = direct_connection
            check[count] = True
            count = count + 1
            if betterconnection:
                new_connection_merged = gpd.GeoDataFrame(pd.concat([new_connection_merged, best_connection], sort=True))
            elif not betterconnection:
                if grid_resume.Connection_Length.values[k] > 0:
                    best_connection = gpd.read_file('Connection_Grid' + str(i) + '.shp')
                    new_connection_merged = gpd.GeoDataFrame(
                        pd.concat([new_connection_merged, best_connection], sort=True))

    grid_resume.to_csv('grid_resume_opt.csv')
    new_connection_merged.crs = geodf_in.crs
    new_connection_merged.to_file('total_connections_opt')
    return grid_resume


#  -----------------------------------  IMPORTING THE WEIGHTED CSV-----------------------------------------------------

os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Input/2_weighted_datasets')
file_name = 'enel_studyarea3_weighted'
geo_csv = pd.read_csv(file_name + ".csv")
print("Layer file successfully loaded.")

geometry = [Point(xy) for xy in zip(geo_csv.X, geo_csv.Y)]
Proj_coords = 32723
geodf_in = gpd.GeoDataFrame(geo_csv, crs=from_epsg(Proj_coords), geometry=geometry)
unit = 1
resolution = 1000
paycheck = 10000
pop_threshold = 20
limitHV = 3000
limitMV = 300
extracostHV = 5000000

#  IMPORTING CLUSTERS

os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Output\Clusters')
gdf_clusters = gpd.read_file("gdf_clusters.shp")
clusters_list_2 = np.unique(gdf_clusters['clusters'])  # takes all unique values inside the dataframe
clusters_list_2 = np.delete(clusters_list_2, np.where(clusters_list_2 == -1))  # removes the noises

os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Output\Grids')
grid_resume_old = pd.read_csv('grid_resume.csv')
clusters_load = grid_resume_old['ClusterLoad']
grid_resume = grid_resume_old.sort_values(by='ClusterLoad')
# clusters_list_2 = grid_resume['Cluster']

print("Clusters successfully imported")

#  IMPORTING THE LOWER RESOLUTION GRID

os.chdir('D:\Vinicius\Mestrado\Thesis\QGIS\Study_area3\Resolution')
gdf_zoomed = gpd.read_file("enel_studyarea3_4000.shp")


#  IMPORTING SUBSTATIONS
os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE/Input//0_Dataset//Substations')
substations = gpd.read_file("substations.shp")
substations['ID'] = range(0, len(substations))

os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Output\Chunks')


#  --------------------------------------------------------------------------------------------------------------------

grid_resume = pd.DataFrame(index=clusters_list_2,
                           columns=['Cluster', 'Chunk_Length', 'Chunk_Cost', 'Grid_Length',
                                    'Grid_Cost', 'Connection_Length', 'Connection_Cost',
                                    'ConnectionType', 'Link_Length', 'Link_Cost'])
grid_resume['Cluster'] = clusters_list_2
grid_resume['ClusterLoad'] = grid_resume_old.ClusterLoad.values
total_chunk = total_derivations = pd.DataFrame()
total_branch_steiner = total_branch_spider = chunk_points = all_branch_direct = total_connections = pd.DataFrame()


# ---------------------------------------START OF THE MAIN CHUNK CREATION----------------------------------------------

for i in clusters_list_2:
    gdf_cluster_only = gdf_clusters[gdf_clusters['clusters'] == i]
    gdf_zoomed_pop = gdf_zoomed[gdf_zoomed['Cluster'] == i]

    gdf_zoomed_pop = gdf_zoomed_pop[gdf_zoomed_pop['Population'] >= pop_threshold]

    points_to_electrify = int(len(gdf_zoomed_pop))
    if points_to_electrify > 2:
        print('-' * 100)
        print('CREATING CHUNK FOR CLUSTER ' + str(i))
        print('-' * 100)

        chunk_grid, chunk_cost, chunk_length, connections2 = Steinerman(gdf_cluster_only, gdf_zoomed_pop,
                                                                        Proj_coords, paycheck, resolution)
        fileout = "Chunk_" + str(i) + '.shp'
        grid_resume.at[i, 'Chunk_Length'] = chunk_length / 1000
        grid_resume.at[i, 'Chunk_Cost'] = chunk_cost / 1000
        chunk_grid.to_file(fileout)
        total_chunk = gpd.GeoDataFrame(pd.concat([total_chunk, chunk_grid], sort=True))

    elif points_to_electrify == 2:
        print('-' * 100)
        print('CREATING CHUNK FOR CLUSTER ' + str(i))
        print('-' * 100)
        p1 = gdf_zoomed_pop[gdf_zoomed_pop['ID'] == gdf_zoomed_pop.ID.values[0]]
        p2 = gdf_zoomed_pop[gdf_zoomed_pop['ID'] == gdf_zoomed_pop.ID.values[1]]
        chunk_grid, chunk_cost, chunk_length = grid_direct_connection(geodf_in, p1, p2, Proj_coords, paycheck,
                                                                      resolution)
        fileout = 'Chunk_' + str(i) + '.shp'
        grid_resume.at[i, 'Chunk_Length'] = chunk_length / 1000
        grid_resume.at[i, 'Chunk_Cost'] = chunk_cost / 1000
        chunk_grid.to_file(fileout)
        total_chunk = gpd.GeoDataFrame(pd.concat([total_chunk, chunk_grid], sort=True))

    elif points_to_electrify < 2:
        print('-' * 100)
        print('NO CHUNK NECESSARY FOR CLUSTER ' + str(i))
        print('-' * 100)
        fileout = 'Chunk_' + str(i) + '.shp'
        grid_resume.at[i, 'Chunk_Length'] = 0
        grid_resume.at[i, 'Chunk_Cost'] = 0
total_chunk.crs = geodf_in.crs
total_chunk.to_file('total_chunk')

#  -------------------------------------START OF THE BRANCH TECHNIQUES-------------------------------------------------

pop_threshold = 1
paycheck = 3333
for i in clusters_list_2:

    gdf_clusters_pop = gdf_clusters[gdf_clusters['clusters'] == i]
    gdf_clusters_pop = gdf_clusters_pop[gdf_clusters_pop['Population'] >= pop_threshold]

    # --------------------------------- ASSIGNING SUBSTATIONS ----------------------------------------------------------
    substations_new = substations
    if clusters_load[np.where(clusters_list_2 == i)[0][0]] > limitHV:
        substations_new = substations[substations['Type'] == 'HV']
    elif clusters_load[np.where(clusters_list_2 == i)[0][0]] > limitMV and clusters_load[
        np.where(clusters_list_2 == i)[0][0]] < limitHV:
        substations_new = substations[substations['Type'] == 'MV_3P']
    elif clusters_load[np.where(clusters_list_2 == i)[0][0]] < limitMV:
        substations_new = substations[substations['Type'] != 'HV']

    unary_union = geodf_in.unary_union
    substations_new['nearest_id'] = substations_new.apply(nearest, geom_union=unary_union, df1=substations_new,
                                                          df2=geodf_in,
                                                          geom1_col='geometry', src_column='ID', axis=1)
    subinmesh = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
    for k, row in substations_new.iterrows():
        subinmesh = subinmesh.append(geodf_in[geodf_in['ID'] == row['nearest_id']], sort=False)
    subinmesh.reset_index(drop=True, inplace=True)

    d1 = {'x': gdf_clusters_pop['X'], 'y': gdf_clusters_pop['Y'], 'z': gdf_clusters_pop['Elevation']}
    cluster_loc = pd.DataFrame(data=d1)
    cluster_loc.index = gdf_clusters_pop['ID']

    d2 = {'x': subinmesh['X'], 'y': subinmesh['Y'], 'z': subinmesh['Elevation']}
    sub_loc = pd.DataFrame(data=d2)
    sub_loc.index = subinmesh['ID']
    Distance_3D = pd.DataFrame(cdist(cluster_loc.values, sub_loc.values, 'euclidean'), index=cluster_loc.index,
                               columns=sub_loc.index)
    mm = Distance_3D.min().idxmin()
    substation_designata = subinmesh[subinmesh['ID'] == mm]
    uu_substation = substation_designata.unary_union
    connectiontype = substations_new[substations_new['nearest_id'] == mm]
    grid_resume.at[i, 'ConnectionType'] = connectiontype.Type.values[0]

    # ------CHECKING IF THERE IS A CHUNK ON THIS CLUSTER, IF NOT JUST RUN SPIDERMAN AND STEINERMAN--------------------

    if grid_resume.at[i, 'Chunk_Length'] == 0:
        print('-' * 100)
        print('CREATING GRID FOR CLUSTER ' + str(i))
        print('-' * 100)
        chunk_grid, chunk_cost, chunk_length, connections2 = Steinerman(geodf_in, gdf_clusters_pop,
                                                                        Proj_coords, paycheck, resolution)
        fileout = "Chunk_" + str(i) + '.shp'
        chunk_grid.to_file(fileout)
        grid_resume.at[i, 'Grid_Length'] = chunk_length / 1000
        grid_resume.at[i, 'Grid_Cost'] = chunk_cost / 1000
        total_branch_steiner = gpd.GeoDataFrame(pd.concat([total_branch_steiner, chunk_grid], sort=True))

        uu_chunk = chunk_grid.unary_union
        nearpoints = nearest_points(uu_chunk, uu_substation)
        p1 = gdf_clusters[gdf_clusters['geometry'] == nearpoints[0]]
        if p1.ID.values[0] == substation_designata.ID.values[0]:
            grid_resume.at[i, 'Connection_Length'] = 0
            grid_resume.at[i, 'Connection_Cost'] = 0
            continue
        connection_grid, connection_cost, connection_length = grid_direct_connection(geodf_in, p1, substation_designata,
                                                                                     Proj_coords,
                                                                                     paycheck,
                                                                                     resolution)

        total_connections = gpd.GeoDataFrame(pd.concat([total_connections, connection_grid], sort=True))
        fileout = "Connection_Grid" + str(i) + '.shp'
        connection_grid.to_file(fileout)
        grid_resume.at[i, 'Connection_Length'] = connection_length / 1000
        if connectiontype.Type.values[0] == 'HV':
            grid_resume.at[i, 'Connection_Cost'] = connection_cost / 1000 + extracostHV / 1000
        else:
            grid_resume.at[i, 'Connection_Cost'] = connection_cost / 1000
        # chunk_grid, chunk_cost, chunk_length, connections2 = Spiderman(geodf_in, gdf_clusters_pop,
        #                                                                 Proj_coords, paycheck, resolution)
        # fileout = "Spider_Grid" + str(i) + '.shp'
        # chunk_grid.to_file(fileout)
        # grid_resume.at[i, 'Branch_Length_Spiderman'] = chunk_length/1000
        # grid_resume.at[i, 'Branch_Cost_Spiderman'] = chunk_cost/1000
        # total_branch_spider = gpd.GeoDataFrame(pd.concat([total_branch_spider, chunk_grid], sort=True))
        continue

    #--------  ASSIGNING WEIGHT EQUAL TO 0 TO THE MAIN CHUNK AND PERFORMING SPIDERMAN AND STEINERMAN-------------------

    chunk = gpd.read_file("Chunk_" + str(i) + ".shp")
    print('-' * 100)
    print('CREATING DERIVATION FOR CLUSTER ' + str(i))
    print('-' * 100)
    geodf_noweight = geodf_in
    for id in chunk.ID1:
        geodf_in_id = geodf_in[geodf_in['ID'] == id]
        geodf_in_id.loc[:, 'Weight'] = 0.001
        geodf_noweight.loc[np.where(geodf_in.ID == id)[0][0], 'Weight'] = 0.001
    last_id = chunk.loc[chunk.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
    geodf_in_id = geodf_in[geodf_in['ID'] == last_id]
    geodf_in_id.loc[:, 'Weight'] = 0.001
    geodf_noweight[np.where(geodf_in.ID == last_id)[0][0], 'Weight'] = 0.001

    chunk_grid, chunk_cost, chunk_length, connections2 = Steinerman(geodf_noweight, gdf_clusters_pop,
                                                                    Proj_coords, paycheck, resolution)
    count = 0

    for value in chunk_grid.values:
        if value[2] in chunk.values and value[3] in chunk.values:
            chunk_grid = chunk_grid.drop((np.where(chunk_grid.values[:, 1] == value[1])[0][0]))
            chunk_grid = chunk_grid.reset_index(drop=True)
    # for value in chunk_grid.values:  # this computes the cost of the grid reassigning the weights to the original value
    #     weight1 = gdf_clusters[gdf_clusters['ID'] == value[2]].Weight.values[0]
    #     weight2 = gdf_clusters[gdf_clusters['ID'] == value[3]].Weight.values[0]
    #     value[1] = ((weight1 + weight2) / 2) * paycheck * value[5] / 1000
    #     chunk_grid.Weight.values[count] = value[1]
    #     count = count + 1
    chunk_cost = sum(chunk_grid['Weight'])
    grid_resume.at[i, 'Grid_Cost'] = chunk_cost / 1000
    grid_resume.at[i, 'Grid_Length'] = chunk_length / 1000
    fileout = "Steiner_Grid" + str(i) + '.shp'
    chunk_grid.to_file(fileout)
    total_branch_steiner = gpd.GeoDataFrame(pd.concat([total_branch_steiner, chunk_grid], sort=True))
    chunk_inmesh = pd.DataFrame()
    for id in chunk.ID1:  # getting all the points inside the chunk to compute the nearest_points
        chunk_inmesh = gpd.GeoDataFrame(pd.concat([chunk_inmesh, gdf_clusters[gdf_clusters['ID'] == id]], sort=True))
    last_id = chunk.loc[chunk.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
    chunk_inmesh = gpd.GeoDataFrame(pd.concat([chunk_inmesh, gdf_clusters[gdf_clusters['ID'] == last_id]], sort=True))
    uu_chunk = chunk_inmesh.unary_union
    nearpoints = nearest_points(uu_chunk, uu_substation)
    p1 = gdf_clusters[gdf_clusters['geometry'] == nearpoints[0]]
    if p1.ID.values[0] == substation_designata.ID.values[0]:
        grid_resume.at[i, 'Connection_Length'] = 0
        grid_resume.at[i, 'Connection_Cost'] = 0
        continue

    connection_grid, connection_cost, connection_length = grid_direct_connection(geodf_in, p1, substation_designata,
                                                                                 Proj_coords, paycheck,
                                                                                 resolution)

    total_connections = gpd.GeoDataFrame(pd.concat([total_connections, connection_grid], sort=True))
    fileout = "Connection_Grid" + str(i) + '.shp'
    connection_grid.to_file(fileout)
    grid_resume.at[i, 'Connection_Length'] = connection_length / 1000
    if connectiontype.Type.values[0] == 'HV':
        grid_resume.at[i, 'Connection_Cost'] = connection_cost / 1000 + extracostHV / 1000
    else:
        grid_resume.at[i, 'Connection_Cost'] = connection_cost / 1000

    # chunk_grid, chunk_cost, chunk_length, connections2 = Spiderman(geodf_in, gdf_clusters_pop,
    #                                                                Proj_coords, paycheck, resolution)
    # count = 0
    # for value in chunk_grid.values:
    #     weight1 = gdf_clusters[gdf_clusters['ID'] == value[0]].Weight.values[0]
    #     weight2 = gdf_clusters[gdf_clusters['ID'] == value[1]].Weight.values[0]
    #     value[2] = ((weight1 + weight2) / 2) * paycheck * value[4] / 1000
    #     chunk_grid.Weight.values[count] = value[2]
    #     count = count + 1
    # chunk_cost = sum(chunk_grid['Weight'])
    # grid_resume.at[i, 'Branch_Cost_Spiderman'] = chunk_cost / 1000
    # grid_resume.at[i, 'Branch_Length_Spiderman'] = chunk_length / 1000
    # fileout = "Spider_Grid" + str(i) + '.shp'
    # chunk_grid.to_file(fileout)
    # total_branch_spider = gpd.GeoDataFrame(pd.concat([total_branch_spider, chunk_grid], sort=True))

# -----------------------------START OF THE DIRECT CONNECTION TECHNIQUE-------------------------------------------
#
#     total_branch_direct = pd.DataFrame()
#     total_cost = 0
#     total_length = 0
#
#     for id in chunk.ID1:  #getting all the points inside the chunk to compute the nearest_points
#         chunk_points = gpd.GeoDataFrame(pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == id]], sort=True))
#         last_id = chunk.loc[chunk.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
#         chunk_points = gpd.GeoDataFrame(pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == last_id]], sort=True))
#     uu_chunk = chunk_points.unary_union
#
#     for geo in gdf_clusters_pop.geometry:
#         new_chunk = chunk_points
#         uu_chunk_new = uu_chunk
#         best_cost = 999999999
#         for count in range(3):
#             nearpoints = nearest_points(uu_chunk_new, geo)
#             if nearpoints[0].distance(nearpoints[1]) < resolution*0.3:
#                 best_branch = pd.DataFrame()
#                 best_length = 0
#                 best_cost = 0
#                 continue
#             p1 = gdf_clusters[gdf_clusters['geometry'] == nearpoints[0]]
#             p2 = gdf_clusters[gdf_clusters['geometry'] == nearpoints[1]]
#             blockPrint()
#             branch_grid, branch_cost, branch_length = grid_direct_connection(geodf_in, p1, p2, Proj_coords, paycheck, resolution)
#             if branch_cost < best_cost:
#                 best_cost = branch_cost
#                 best_branch = branch_grid
#                 best_length = branch_length
#             enablePrint()
#             dropid = chunk_points[chunk_points['geometry'] == nearpoints[0]].index
#             new_chunk = new_chunk.drop([dropid.values[0]])
#             uu_chunk_new = new_chunk.unary_union
#
#         total_branch_direct = gpd.GeoDataFrame(pd.concat([total_branch_direct, best_branch], sort=True))
#         total_length = total_length + best_length
#         total_cost = total_cost + best_cost
#
#     #  REMOVING DUPLICATIONS
#     uniqueID = np.unique(total_branch_direct.ID1, return_index=True)
#     total_branch_direct['Index'] = np.arange(total_branch_direct.values.__len__()).reshape(total_branch_direct.values.__len__(),1)
#     total_branch_direct_nodups = pd.DataFrame()
#     for id in uniqueID[1]:
#         total_branch_direct_nodups = gpd.GeoDataFrame(pd.concat([total_branch_direct_nodups, total_branch_direct[total_branch_direct['Index'] == id]], sort=True))
#
#     total_branch_direct_nodups.crs = geodf_in.crs
#     fileout = 'branch_direct' + str(i) + '.shp'
#     total_branch_direct_nodups.to_file(fileout)
#     grid_resume.at[i, 'Branch_Length_Direct'] = total_length/1000
#     grid_resume.at[i, 'Branch_Cost_Direct'] = total_cost/1000
#     all_branch_direct = gpd.GeoDataFrame(pd.concat([all_branch_direct, total_branch_direct_nodups], sort=True))



#  -------------------------------- CONNECTION OF POINTS OUTSIDE THE CLUSTERS ----------------------------------------

total_grid = total_branch_steiner
total_grid.reset_index(inplace=True, drop=True)
total_link = grid_points = total_link_nodups = pd.DataFrame()
gdf_clusters_out = gdf_clusters[gdf_clusters['clusters'] == -1]
gdf_clusters_out = gdf_clusters_out[gdf_clusters_out['Population'] >= 1]

for id in total_grid.ID1:  #getting all the points inside the chunk to compute the nearest_points
    grid_points = gpd.GeoDataFrame(pd.concat([grid_points, gdf_clusters[gdf_clusters['ID'] == id]], sort=True))
last_id = total_grid.loc[total_grid.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
grid_points = gpd.GeoDataFrame(pd.concat([grid_points, gdf_clusters[gdf_clusters['ID'] == last_id]], sort=True))
uu_grid = grid_points.unary_union


for i in gdf_clusters_out.geometry:
    nearpoints = nearest_points(uu_grid, i)
    p1 = gdf_clusters[gdf_clusters['geometry'] == nearpoints[0]]
    p2 = gdf_clusters[gdf_clusters['geometry'] == nearpoints[1]]
    blockPrint()
    link, link_cost, link_length = grid_direct_connection(geodf_in, p1, p2, Proj_coords, paycheck, resolution)
    enablePrint()
    total_link = gpd.GeoDataFrame(pd.concat([total_link, link], sort=True))

    #  REMOVING DUPLICATIONS
uniqueID = np.unique(total_link.ID1, return_index=True)
total_link['Index'] = np.arange(total_link.values.__len__()).reshape(total_link.values.__len__(),1)
total_link_nodups = pd.DataFrame()
for id in uniqueID[1]:
    total_link_nodups = gpd.GeoDataFrame(pd.concat([total_link_nodups, total_link[total_link['Index'] == id]], sort=True))
link_cost = total_link_nodups.Weight.sum()
link_length = total_link_nodups.length.sum()
grid_resume.at[0, 'Link_Length'] = link_length/1000
grid_resume.at[0, 'Link_Cost'] = link_cost/1000




total_branch_steiner.crs = total_branch_spider.crs = all_branch_direct.crs = total_connections.crs = total_link_nodups.crs = geodf_in.crs
total_branch_steiner.to_file('total_branch_steiner')
# total_branch_spider.to_file('total_branch_spider')
# all_branch_direct.to_file('total_branch_direct')
total_connections.to_file('total_connections')
total_link_nodups.to_file('total_link')

# for i in grid_resume:
#     if i == 'ConnectionType':
#         continue
#     grid_resume.at[len(clusters_list_2) + 1, i] = 0
#     for j in range(len(clusters_list_2)):
#         grid_resume.at[len(clusters_list_2) + 1, i] = grid_resume.at[len(clusters_list_2) + 1, i] + \
#                                                       grid_resume[i].values[j]

grid_resume.to_csv('grid_resume.csv')
grid_optimized = grid_optimization(gdf_clusters, geodf_in, grid_resume, Proj_coords, resolution, paycheck)


# ------------------------------------- SIZING EACH DERIVATION --------------------------------------------------------
for cluster in clusters_list_2:
    if grid_resume.at[cluster, 'Chunk_Length'] == 0:
        continue
    derivations = gpd.read_file('Steiner_Grid' + str(cluster) + '.shp')
    chunk = gpd.read_file('Chunk_' + str(cluster) + '.shp')
    for i in derivations.values:
        if i[2] in chunk.values and i[3] in chunk.values:
            derivations = derivations.drop((np.where(derivations.values[:, 1] == i[1])[0][0]))
            derivations = derivations.reset_index(drop=True)
    count = 1
    G = nx.Graph()
    G.add_edges_from([*zip(list(derivations.ID1),list(derivations.ID2))])
    components = list(nx.algorithms.components.connected_components(G))
    for i in components:
        points = np.array(list(i))
        pop_derivation = 0
        for j in points:
            for k in range(derivations.values.__len__()):
                if derivations.at[k,'ID1'] == j or derivations.at[k,'ID2'] == j:
                    derivations.at[k, 'id'] = count
            pop_derivation = pop_derivation + gdf_clusters[gdf_clusters['ID'] == j].Population.values
        derivations.loc[derivations['id'] == count, 'Power'] = int(pop_derivation)*0.4
        count = count+1
    total_derivations = gpd.GeoDataFrame(pd.concat([total_derivations, derivations], sort=True))
total_derivations.crs = geodf_in.crs
total_derivations.to_file('derivations')



