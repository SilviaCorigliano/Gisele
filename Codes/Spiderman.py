import math
import time
import pandas as pd
import networkx as nx
from scipy import sparse
from supporting_GISEle2 import *
from Codes import dijkstra


def spider(geo_df, gdf_cluster_pop, line_bc, resolution):
    print('Running spider algorithm..')
    start_time = time.time()

    gdf_cluster_pop.index = pd.Series(range(0, len(gdf_cluster_pop['ID'])))
    dist_3d_matrix = distance_3d(gdf_cluster_pop, gdf_cluster_pop, 'X', 'Y',
                                 'Elevation')
    dist_2d_matrix = distance_2d(gdf_cluster_pop, gdf_cluster_pop, 'X', 'Y')
    edges_matrix = weight_matrix(gdf_cluster_pop, dist_3d_matrix, line_bc)
    length_limit = 99999999
    edges_matrix[dist_2d_matrix > math.ceil(length_limit)] = 0
    edges_matrix_sparse = sparse.csr_matrix(edges_matrix)
    graph = nx.from_scipy_sparse_matrix(edges_matrix_sparse)

    tree = nx.minimum_spanning_tree(graph, weight='weight')
    path = list(tree.edges)
    c_grid, c_grid_points = edges_to_line(path, gdf_cluster_pop, edges_matrix)
    c_grid['Length'] = c_grid.length.astype(int)

    length_limit = resolution*1.5
    long_lines = c_grid.loc[c_grid['Length'] > math.ceil(length_limit)]
    short_lines = c_grid.loc[c_grid['Length'] <= math.ceil(length_limit)]
    short_lines = short_lines.reset_index(drop=True)
    print('Number of long lines: ', long_lines.__len__())
    print('Number of short lines: ', short_lines.__len__())

    if len(long_lines) != 0:

        long_lines = long_lines.sort_values(by='Length', ascending=True)
        long_lines = long_lines.reset_index(drop=True)
        for i, row in long_lines.iterrows():
            point_1 = gdf_cluster_pop[gdf_cluster_pop['ID'] == row.ID1]
            point_2 = gdf_cluster_pop[gdf_cluster_pop['ID'] == row.ID2]
            c_grid_points = list(zip(short_lines.ID1, short_lines.ID2))

            segment, segment_cost, segment_length = \
                dijkstra.dijkstra_connection(geo_df, point_1, point_2,
                                             c_grid_points, line_bc,
                                             resolution)

            short_lines = pd.concat([short_lines, segment], sort=True)
            short_lines = short_lines.reset_index(drop=True)

    c_grid = short_lines
    c_grid['Length'] = c_grid.length.astype(int)
    c_grid_cost = int(c_grid['Cost'].sum(axis=0))
    c_grid_length = c_grid['Length'].sum(axis=0)
    c_grid_points = list(zip(c_grid.ID1, c_grid.ID2))

    final_time = time.time()
    total_time = final_time - start_time
    print("The total time required by the Spider algorithm was: "
          + str(round(total_time, 4)) + " seconds")

    return c_grid, c_grid_cost, c_grid_length, c_grid_points
