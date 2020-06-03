# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:27:23 2019

@author: Silvia
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
import time
import math
from scipy import sparse
from fiona.crs import from_epsg
from scipy.spatial import distance_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from Codes.Weight_matrix import*
from Codes.Steiner_tree_code import*
from shapely.geometry import Point, LineString, Polygon, box
from scipy.spatial.distance import cdist


def Steinerman(mesh, cluster_points, Proj_coords, paycheck, resolution):

    start_time = time.time()
    # remove repetitions in the dataframe
    mesh.drop_duplicates(['ID'], keep='last', inplace=True)
    # cluster_points.drop_duplicates(['ID'], keep='last', inplace=True)
    # Creation of a pandas DataFrame containing only coordinates of populated points
    d = {'x': cluster_points['X'], 'y': cluster_points['Y'], 'z': cluster_points['Elevation']}
    pop_points = pd.DataFrame(data=d)
    pop_points.index = cluster_points['ID']
    print('pop_points has been created')

    # Building a box
    xmin = cluster_points['X'].min()
    xmax = cluster_points['X'].max()
    ymin = cluster_points['Y'].min()
    ymax = cluster_points['Y'].max()
    wide = Point(xmin, ymin).distance(Point(xmax, ymax))
    extension = wide / 5
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
    diag_length = resolution * 1.5

    # Definition of weighted connections matrix
    Edges_matrix = Weight_matrix
    Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 0
    Edges_matrix_sparse = sparse.csr_matrix(Edges_matrix)
    # Edges_matrix = Edges_matrix[Distance_2D_box > math.ceil(diag_length)]
    print('Checkpoint 4: Edges matrix completed')
    Graph = nx.from_scipy_sparse_matrix(Edges_matrix_sparse)
    # Graph.remove_edges_from(np.where(Edges_matrix.to_numpy() == 999999))

    # Lavoro per reperire i giusti indici di cluster_points
    a = list(cluster_points['ID'].to_numpy())
    cluster_points_box = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
    for i in a:
        p = df_box.loc[df_box['ID'] == i]
        cluster_points_box = pd.concat([cluster_points_box, p], sort=True)

    # Steiner tree connecting point with population over a defined threshold
    terminal_nodes = list(cluster_points_box.index)
    T = steiner_tree(Graph, terminal_nodes, weight='weight')
    # create output shapefile
    Adj_matrix= nx.to_numpy_matrix(T)
    N = (Adj_matrix > 0).sum()
    nodes = list(T.nodes)
    Adj_matrix = pd.DataFrame(Adj_matrix)
    Adj_matrix.columns = nodes
    Adj_matrix.index = nodes
    # Adj_matrix[Adj_matrix==9999]=0

    connections = []
    k = 0
    grid = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
    print('Checkpoint 5')
    #grid['T']=()
    grid['id'] = pd.Series(range(1, N + 1))
    grid['Weight'] = pd.Series(range(1, N + 1))
    grid['ID1'] = pd.Series(range(1, N + 1))
    grid['ID2'] = pd.Series(range(1, N + 1))
    grid.loc[:, 'geometry'] = None
    # grid['geometry'] = None
    grid['geometry'].astype = gpd.geoseries.GeoSeries
    for i, row in Adj_matrix.iterrows():
        for j, column in row.iteritems():
            con = [min(df_box.loc[i, 'ID'], df_box.loc[j, 'ID']),
                   max(df_box.loc[i, 'ID'], df_box.loc[j, 'ID'])]
            if column > 0 and con not in connections:
                grid.at[k, ['geometry']] = LineString([(df_box.loc[i, ['X']], df_box.loc[i, ['Y']]),
                                                       (df_box.loc[j, ['X']], df_box.loc[j, ['Y']])])
                grid.at[k, ['id']] = 0
                grid.at[k, ['Weight']] = Edges_matrix.loc[df_box.loc[i, 'ID'], df_box.loc[j, 'ID']]
                grid.at[k, 'ID1'] = df_box.loc[i, 'ID']
                grid.at[k, 'ID2'] = df_box.loc[j, 'ID']
                connections.append(con)
                k = k+1
    grid.head()
    indexNames = grid[(grid['id'] > 0)].index
    grid.drop(indexNames, inplace=True)
    grid.dropna(inplace=True)
    grid.crs = cluster_points.crs
    total_cost = grid['Weight'].sum(axis=0)
    grid['line_length'] = grid['geometry'].length
    total_length = grid['line_length'].sum(axis=0)

    final_time = time.time()
    total_time = final_time - start_time
    print("Total time for tree's evolution algorithm was: " + str(total_time) + "seconds")

    return grid, total_cost, total_length, connections

