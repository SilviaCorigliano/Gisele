from supporting_GISEle2 import*
import networkx as nx
import math
from scipy import sparse


def dijkstra_connection(geo_df, connecting_point, assigned_substation,
                        c_grid_points, line_bc, resolution):

    connection = gpd.GeoDataFrame()
    connection_cost = 0
    connection_length = 0

    dist = assigned_substation.unary_union.distance(connecting_point.
                                                    unary_union)
    if dist > 50 * resolution:
        print('Connection distance too long to use Dijkstra')
        connection_cost = 999999
        connection_length = 0
        connection = gpd.GeoDataFrame()
        return connection, connection_cost, connection_length

    df_box = create_box(pd.concat([assigned_substation, connecting_point]),
                        geo_df)

    dist_2d_matrix = distance_2d(df_box, df_box, 'X', 'Y')
    dist_3d_matrix = distance_3d(df_box, df_box, 'X', 'Y', 'Elevation')

    if np.any(dist_2d_matrix):

        edges_matrix = weight_matrix(df_box, dist_3d_matrix, line_bc)
        length_limit = resolution * 1.5
        edges_matrix[dist_2d_matrix > math.ceil(length_limit)] = 0

        #  reduces the weights of edges already present in the cluster grid
        for i in c_grid_points:
            if i[0] in edges_matrix.index.values and \
                    i[1] in edges_matrix.index.values:
                edges_matrix.loc[i[0], i[1]] = 0.001
                edges_matrix.loc[i[1], i[0]] = 0.001

        edges_matrix_sparse = sparse.csr_matrix(edges_matrix)
        graph = nx.from_scipy_sparse_matrix(edges_matrix_sparse)
        source = df_box.loc[df_box['ID'] == int(assigned_substation['ID']), :]
        source = int(source.index.values)
        target = df_box.loc[df_box['ID'] == int(connecting_point['ID']), :]
        target = int(target.index.values)

        path = nx.dijkstra_path(graph, source, target, weight='weight')
        steps = len(path)
        new_path = []
        for i in range(0, steps-1):
            new_path.append(path[i+1])

        path = list(zip(path, new_path))

        # Creating the shapefile

        connection, pts = edges_to_line(path, df_box, edges_matrix)

        connection['Length'] = connection.length.astype(int)
        connection_cost = int(connection['Cost'].sum(axis=0))
        connection_length = connection['Length'].sum(axis=0)

    return connection, connection_cost, connection_length
