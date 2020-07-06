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


# def add_roads(gdf_roads, geo_df):
#     line_vertices = pd.DataFrame(columns=['ID', 'X', 'Y', 'ID_line', 'Weight'])
#     #create geodataframe with all the elementary segments that compose the road
#     segments = gpd.GeoDataFrame(columns=['geometry', 'ID1', 'ID2'])
#     k = 0
#     w = geo_df.shape[0]
#     x = 0
#     splitted = []
#     for i, row in gdf_roads.iterrows():
#         for j in list(row['geometry'].coords):
#             line_vertices.loc[w,'X']=j[0]
#             line_vertices.loc[w,'Y']=j[1]
#             line_vertices.loc[w,'ID_line']=k
#             line_vertices.loc[w,'ID']=w
#             line_vertices.loc[w,'Weight']=1
#             w=w+1
#         k=k+1
#         points_to_split = MultiPoint([Point(x,y) for x,y in row['geometry'].coords[1:]])
#         splitted = split(row['geometry'],points_to_split)
#         for i in splitted:
#             #print(i)
#             segments.loc[x,'geometry']=i
#             segments.loc[x,'length']=segments.loc[x,'geometry'].length/1000
#             segments.loc[x,'ID1']=line_vertices[(line_vertices['X']==i.coords[0][0])&(line_vertices['Y']==i.coords[0][1])]['ID'].values[0]
#             segments.loc[x,'ID2']=line_vertices[(line_vertices['X']==i.coords[1][0])&(line_vertices['Y']==i.coords[1][1])]['ID'].values[0]
#             x=x+1
#
#     # append to the grid dataframe the point of the road
#     geo_df = geo_df.append(line_vertices)
#     d = {'x': geo_df['X'], 'y': geo_df['Y']}
#     df = pd.DataFrame(data=d)
#     # 2D distance matrix
#     diag_length = resolution * np.sqrt(2)
#     Distmatrix = distance_matrix(df.values, df.values)
#     Distmatrix[Distmatrix > 1.2 * diag_length] = 0
#     # create weigth matrix with the function of Gisele
#     Weight_matrix = weight_matrix(geo_df, Distmatrix)
#     Weight_matrix = np.array(Weight_matrix,
#                              dtype=float)  # otherwise not working, creating an object
#     Weight_matrix_sparse = sparse.csr_matrix(Weight_matrix)
#     # create graph from sparse matrix
#     G = nx.from_scipy_sparse_matrix(Weight_matrix_sparse)
#
#     # add to the graph also the road segments
#     for x in range(segments.shape[0]):
#         # add edges to the graph, each edge corresponds to a segment of the road, with weigth=distance[km]
#         G.add_edge(segments.loc[x, 'ID1'], segments.loc[x, 'ID2'],
#                    weight=segments.loc[x, 'length'])
#     # Apply Disktra between a source and a target that were defined in QGIS
#     source = geo_df[geo_df['Start'] == 1]['ID'].values[0]
#     target = geo_df[geo_df['Target'] == 1]['ID'].values[0]
#
#     path = nx.dijkstra_path(G, source, 8501, weight='weight')
#     path_length_2 = nx.dijkstra_path_length(G, source, 8501, weight='weight')
#     # assign to the list of nodes in the path their coordinates
#     path_coords = gpd.GeoDataFrame(index=path, columns=['X', 'Y', 'order'])
#     k = 0
#     for i in path:
#         path_coords.loc[i, 'X'] = geo_df.loc[i, 'X']
#         path_coords.loc[i, 'Y'] = geo_df.loc[i, 'Y']
#         path_coords.loc[i, 'order'] = k  # useful in QGIS
#         k = k + 1
#     # save file
#     path_coords.to_csv(
#         r'C:\Users\silvi\OneDrive - Politecnico di Milano\Documents\2018-2019\Corsi Seminari e Convegni\Advanced GIS\Dijsktra_vector.csv')
#     return geo_df