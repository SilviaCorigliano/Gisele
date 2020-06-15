import os
import math
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
from fiona.crs import from_epsg
from supporting_GISEle2 import *
from shapely.geometry import Point, box, LineString
from shapely.ops import nearest_points
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy import sparse
from Codes import Steinerman, Spiderman


def substation_assignment(cluster_n, geo_df, gdf_cluster_pop, substations,
                          clusters_list, limit_hv, limit_mv, crs):
    print('Assigning to the nearest substation..')

    if clusters_list[2][np.where(clusters_list[0] == cluster_n)[0][0]] \
            > limit_hv:
        substations = substations[substations['Type'] == 'HV']
    elif clusters_list[2][
        np.where(clusters_list[0] == cluster_n)[0][0]] > limit_mv and \
            clusters_list[2][np.where(clusters_list[0] == cluster_n)[0][0]] \
            < limit_hv:
        substations = substations[substations['Type'] == 'MV']
    elif clusters_list[2][
            np.where(clusters_list[0] == cluster_n)[0][0]] < limit_mv:
        substations = substations[substations['Type'] != 'HV']

    substations = substations.assign(
        nearest_id=substations.apply(nearest, df=geo_df, src_column='ID',
                                     axis=1))

    sub_in_df = gpd.GeoDataFrame(crs=from_epsg(crs))
    for i, row in substations.iterrows():
        sub_in_df = sub_in_df.append(
            geo_df[geo_df['ID'] == row['nearest_id']], sort=False)
    sub_in_df.reset_index(drop=True, inplace=True)

    dist = distance_3d(gdf_cluster_pop, sub_in_df, 'X', 'Y', 'Elevation')

    assigned_substation = sub_in_df[sub_in_df['ID'] == dist.min().idxmin()]
    connection_type = substations[substations['nearest_id']
                                  == dist.min().idxmin()].Type.values[0]

    print('Substation assignment complete.')

    return assigned_substation, connection_type


def cluster_grid(geo_df, gdf_cluster_pop, crs, resolution, line_bc,
                 points_to_electrify):

    if points_to_electrify < 3:
        c_grid = gdf_cluster_pop
        c_grid_cost = 0
        c_grid_length = 0

    elif points_to_electrify < 1000:
        c_grid1, c_grid_cost1, c_grid_length1, c_grid_points1 = Steinerman.\
            Steinerman(geo_df, gdf_cluster_pop, crs, line_bc, resolution)
        c_grid2, c_grid_cost2, c_grid_length2, c_grid_points2 = Spiderman.\
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
        c_grid, c_grid_cost, c_grid_length, c_grid_points = Spiderman.\
            Spiderman(geo_df, gdf_cluster_pop, crs, line_bc, resolution)

    return c_grid, c_grid_cost, c_grid_length, c_grid_points


def substation_connection(c_grid, assigned_substation, c_grid_points, geo_df,
                          line_bc, resolution, crs):
    # c_nodes = list(c_grid['ID1'].values) + list(
    #     c_grid['ID2'].values)
    # c_nodes = list(dict.fromkeys(c_nodes))
    # c_nodes_df = gpd.GeoDataFrame(crs=geo_df.crs)
    # for i in c_nodes:
    #     c_nodes_df = c_nodes_df.append(
    #         geo_df[geo_df['ID'] == i], sort=False)
    # c_nodes_df.reset_index(drop=True, inplace=True)

    c_nodes_df = line_to_points(c_grid, geo_df)

    assigned_substation = assigned_substation.assign(
        nearest_gridpoint=assigned_substation.apply(nearest, df=c_nodes_df,
                                                    src_column='ID', axis=1))

    # ------------------------------------------------------------------------
    point_sub = Point(assigned_substation['X'], assigned_substation['Y'])
    id_grid = int(assigned_substation['nearest_gridpoint'].values)
    point_grid = Point(float(geo_df[geo_df['ID'] == id_grid].X),
                      float(geo_df[geo_df['ID'] == id_grid].Y))
    dist = point_sub.distance(point_grid)
    p1 = geo_df[geo_df['ID'] == id_grid]
    if dist < 30000:
        connection, connection_cost, connection_length = \
            grid_direct_connection(geo_df, assigned_substation, p1, crs,
                                   line_bc, resolution)
    else:
        print('Cluster too far from the main grid to be connected, a'
              ' microgrid solution is suggested')
        connection_cost = 999999
        connection_length = 0
        connection = gpd.GeoDataFrame()
    return connection, connection_cost, connection_length


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
    if np.any(Distance_2D_box):
        Distance_3D_box = pd.DataFrame(
            cdist(Battlefield3.values, Battlefield3.values, 'euclidean'),
            index=Battlefield3.index, columns=Battlefield3.index)
        Weight_matrix = weight_matrix(df_box, Distance_3D_box, paycheck)
        print('3D distance matrix has been created')
        # min_length = Distance_2D_box[Distance_2D_box > 0].min()
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
    elif not np.any(Distance_2D_box):
        cordone_ombelicale = gpd.GeoDataFrame()
        total_cost = 0
        total_length = 0
    return cordone_ombelicale, total_cost, total_length


def routing(geo_df_clustered, geo_df, crs, clusters_list, resolution,
            pop_threshold, input_sub, line_bc, limit_hv, limit_mv):
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
    substations = pd.read_csv(input_sub + '.csv')
    geometry = [Point(xy) for xy in zip(substations['X'], substations['Y'])]
    substations = gpd.GeoDataFrame(substations, geometry=geometry,
                                   crs=from_epsg(crs))
    os.chdir(r'..')
    os.chdir(r'Output//Grids')

    for cluster_n in clusters_list[0]:
        gdf_cluster = geo_df_clustered[
            geo_df_clustered['clusters'] == cluster_n]
        gdf_cluster_pop = gdf_cluster[
            gdf_cluster['Population'] >= pop_threshold]

        points_to_electrify = int(len(gdf_cluster_pop))

        if points_to_electrify > 0:

            l()
            print("Creating grid for Cluster n." + str(cluster_n) + " of "
                  + str(len(clusters_list[0])))
            l()

            assigned_substation, connection_type = \
                substation_assignment(cluster_n, geo_df, gdf_cluster_pop,
                                      substations, clusters_list, limit_hv,
                                      limit_mv, crs)
            grid_resume.loc[cluster_n, 'ConnectionType'] = connection_type

            c_grid, c_grid_cost, c_grid_length, c_grid_points = \
                cluster_grid(geo_df, gdf_cluster_pop, crs, resolution,
                             line_bc, points_to_electrify)

            connection, connection_cost, connection_length = \
                substation_connection(c_grid, assigned_substation,
                                      c_grid_points,
                                      geo_df, line_bc, resolution, crs)

            fileout = 'Grid_' + str(cluster_n) + '.shp'
            c_grid.to_file(fileout)
            grid_merged = gpd.GeoDataFrame(
                pd.concat([grid_merged, c_grid], sort=True))
            fileout = 'connection_' + str(cluster_n) + '.shp'

            if not connection.empty:
                connection.to_file(fileout)
                connection_merged = gpd.GeoDataFrame(pd.concat(
                    [connection_merged, connection], sort=True))

            grid_resume.at[cluster_n, 'Grid_Length'] = c_grid_length / 1000
            grid_resume.at[cluster_n, 'Grid_Cost'] = c_grid_cost / 1000
            grid_resume.at[
                cluster_n, 'Connection_Length'] = connection_length / 1000
            grid_resume.at[
                cluster_n, 'Connection_Cost'] = connection_cost / 1000

        elif points_to_electrify == 0:

            grid_resume.at[cluster_n, 'Grid_Length'] = 0
            grid_resume.at[cluster_n, 'Grid_Cost'] = 0
            grid_resume.at[cluster_n, 'Connection_Length'] = 0
            grid_resume.at[cluster_n, 'Connection_Cost'] = 0

    grid_resume.to_csv('grid_resume.csv')
    grid_merged.crs = connection_merged.crs = geo_df.crs
    grid_merged.to_file('merged_grid')
    connection_merged.to_file('merged_connections')
    os.chdir(r'..//..')
    print("All grids have been optimized and exported.")
    l()
    return grid_resume


def connection_optimization(gdf_clusters, geodf_in, grid_resume, proj_coords,
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

