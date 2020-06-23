import pandas as pd
import geopandas as gpd
import networkx as nx
import time
import math
from scipy import sparse
from supporting_GISEle2 import*
from Codes.Steiner_tree_code import*


def steiner(geo_df, gdf_cluster_pop, line_bc, resolution, branch_points=None):

    if branch_points is None:
        branch_points = []

    print("Running the Steiner's tree algorithm..")

    start_time = time.time()

    #  creating the box around the cluster and calculating edges matrix
    df_box = create_box(gdf_cluster_pop, geo_df)
    dist_2d_matrix = distance_2d(df_box, df_box, 'X', 'Y')
    dist_3d_matrix = distance_3d(df_box, df_box, 'X', 'Y', 'Elevation')

    edges_matrix = weight_matrix(df_box, dist_3d_matrix, line_bc)
    length_limit = resolution * 1.5
    edges_matrix[dist_2d_matrix > math.ceil(length_limit)] = 0

    for i in branch_points:
        if i[0] in edges_matrix.index.values and \
                i[1] in edges_matrix.index.values:
            edges_matrix.loc[i[0], i[1]] = 0.001
            edges_matrix.loc[i[1], i[0]] = 0.001

    edges_matrix_sparse = sparse.csr_matrix(edges_matrix)
    graph = nx.from_scipy_sparse_matrix(edges_matrix_sparse)

    # taking all cluster points inside the box (terminal nodes)
    gdf_cluster_in_box = gpd.GeoDataFrame(crs=geo_df.crs)
    for i in gdf_cluster_pop.ID:
        point = df_box.loc[df_box['ID'] == i]
        gdf_cluster_in_box = pd.concat([gdf_cluster_in_box, point], sort=True)
    terminal_nodes = list(gdf_cluster_in_box.index)

    tree = steiner_tree(graph, terminal_nodes, weight='weight')
    path = list(tree.edges)

    # Creating the shapefile
    c_grid, c_grid_points = edges_to_line(path, df_box, edges_matrix)
    c_grid_cost = int(c_grid['Cost'].sum(axis=0))
    c_grid['Length'] = c_grid.length.astype(int)
    c_grid_length = c_grid['Length'].sum(axis=0)

    final_time = time.time()
    total_time = final_time - start_time
    print("The total time required by the Steiner's tree algorithm was: "
          + str(round(total_time, 4)) + " seconds")

    return c_grid, c_grid_cost, c_grid_length, c_grid_points
