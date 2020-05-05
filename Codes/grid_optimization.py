# -*- coding: utf-8 -*-
"""
Created on Wed Jan 08 2020

@author: Vinicius Gadelha
"""
import glob
import os

import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import time
import math
from fiona.crs import from_epsg
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from Codes.Weight_matrix import *
from Codes.Steiner_tree_code import *
from shapely.geometry import Point, LineString, Polygon, box
from scipy.spatial.distance import cdist
from shapely.ops import nearest_points


def l():
    print('-' * 100)


def grid_direct_connection(mesh, p1, p2, Proj_coords, paycheck, resolution):
    print('Only 1 point to electrify, no internal cluster grid necessary. Connecting to the nearest substation')
    Point_1 = Point(float(p1['X']), float(p1['Y']))
    ID_1 = int(p1['ID'].values)

    Point_2 = Point(float(p2['X']), float(p2['Y']))
    ID_2 = int(p2['ID'].values)
    dist = Point_1.distance(Point_2)

    if dist < 1000:
        extension = dist
    elif dist < 2000:
        extension = dist / 2
    else:
        extension = dist / 4

    xmin = min(Point_1.x, Point_2.x)
    xmax = max(Point_1.x, Point_2.x)
    ymin = min(Point_1.y, Point_2.y)
    ymax = max(Point_1.y, Point_2.y)
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
    min_length = Distance_2D_box[Distance_2D_box > 0].min()
    diag_length = resolution * 1.5

    # Definition of weighted connections matrix
    Edges_matrix = Weight_matrix
    Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 999999999
    print('Checkpoint 4: Edges matrix completed')
    Graph = nx.from_numpy_matrix(Edges_matrix.to_numpy())
    #    Graph.remove_edges_from(np.where(Edges_matrix.to_numpy() == 999999999))
    source = df_box.loc[df_box['ID'].values == ID_1, :]
    target = df_box.loc[df_box['ID'].values == ID_2, :]
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


# IMPORTING THE LINES
l()
print("Starting the optimization process")
l()

os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Output//Clusters')
gdf_clusters = gpd.read_file("gdf_clusters.shp")
clusters_list_2 = np.unique(gdf_clusters['clusters'])  # takes all unique values inside the dataframe
clusters_list_2 = np.delete(clusters_list_2, np.where(clusters_list_2 == -1))  # removes the noises

grid_resume = pd.DataFrame(index=clusters_list_2, columns=['Cluster', 'Grid_Length', 'Grid_Cost', 'Connection_Length',
                                                           'Connection_Cost'])


def grid_optimization(gdf_clusters, geodf_in, grid_resume, proj_coords, resolution, paycheck):
    os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Output\Grids')
    grid_resume = grid_resume.sort_values(by='Connection_Lenghts')
    clusters_list_2 = grid_resume['Cluster']
    grid_resume = grid_resume.sort_values(by='Cluster')
    check = np.zeros(len(clusters_list_2), dtype=bool)

    count = 0

    while not any(check):  # checks if all variables in check are True
        for i in clusters_list_2:
            k = np.where(grid_resume['Cluster'] == i)  # find the index of the cluster we are analysing
            dist = pd.DataFrame(index=clusters_list_2, columns=['Distance', 'NearestPoint'])
            better_connection = False
            for j in clusters_list_2:
                if grid_resume.Connection_Lenghts.values[
                    k] == 0:  # k is necessary due to cluster merging causing indexes and cluster numbers to not match
                    check[count] = True
                    break
                if i == j:
                    dist.Distance[j] = 9999999
                    continue
                grid1 = gpd.read_file("grid_" + str(j) + ".shp")
                grid2 = gpd.read_file("grid_" + str(i) + ".shp")
                uu_grid1 = grid1.unary_union
                uu_grid2 = grid2.unary_union
                dist.NearestPoint[j] = nearest_points(uu_grid1, uu_grid2)
                dist.Distance[j] = dist.NearestPoint[j][0].distance(dist.NearestPoint[j][1])
                if dist.Distance[j] >= grid_resume.Connection_Lenghts.values[k]:
                    continue
                better_connection = True

            if better_connection:
                dist = dist[dist['Distance'] == min(dist.Distance)]
                cluster_to_connect = dist.Distance.index.values[0]
                check2 = np.where(clusters_list_2 == cluster_to_connect)

                if check[check2]:
                    print('There is a better connection for Cluster ' + str(i))
                    p1 = gdf_clusters[gdf_clusters['geometry'] == dist.NearestPoint.values[0][0]]
                    p2 = gdf_clusters[gdf_clusters['geometry'] == dist.NearestPoint.values[0][1]]

                    Point_1 = Point(float(p1['X']), float(p1['Y']))
                    ID_1 = int(p1['ID'].values)

                    Point_2 = Point(float(p2['X']), float(p2['Y']))
                    ID_2 = int(p2['ID'].values)

                    direct_connection, direct_cost, direct_length = grid_direct_connection(geodf_in, p1, p2,
                                                                                           proj_coords,
                                                                                           paycheck,
                                                                                           resolution)

                    direct = 'NewConnection_' + str(clusters_list_2[count]) + '.shp'
                    direct_connection.to_file(direct)
                    print('A new connection for Cluster ' + str(i) + ' was successfully created')
            check[count] = True
            count = count + 1
    return grid_resume
