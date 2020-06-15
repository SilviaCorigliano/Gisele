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

    nodes = line_to_points(c_grid, geo_df)

    assigned_substation = assigned_substation.assign(
        connecting_point=assigned_substation.apply(nearest, df=nodes,
                                                   src_column='ID', axis=1))

    connecting_point = geo_df[
        geo_df['ID'] == int(assigned_substation['connecting_point'].values)]

    connection, connection_cost, connection_length = \
        dijkstra_connection(geo_df, assigned_substation, connecting_point,
                            crs, line_bc, resolution)

    return connection, connection_cost, connection_length


def dijkstra_connection(geo_df, connecting_point, assigned_substation, crs,
                        line_bc, resolution):

    dist = assigned_substation.unary_union.distance(connecting_point.
                                                    unary_union)
    if dist > 50 * resolution:
        print('Connection distance too long to use Dijkstra')
        connection_cost = 999999
        connection_length = 0
        connection = gpd.GeoDataFrame()
        return connection, connection_cost, connection_length

    print('Running Dijkstra.. ')

    if dist < 1000:
        extension = dist
    elif dist < 2000:
        extension = dist / 2
    else:
        extension = dist / 4

    xmin = min(float(assigned_substation.X), float(connecting_point.X))
    xmax = max(float(assigned_substation.X), float(connecting_point.X))
    ymin = min(float(assigned_substation.Y), float(connecting_point.Y))
    ymax = max(float(assigned_substation.Y), float(connecting_point.Y))

    bubble = box(minx=xmin - extension, maxx=xmax + extension,
                 miny=ymin - extension, maxy=ymax + extension)
    df_box = geo_df[geo_df.within(bubble)]
    df_box.index = pd.Series(range(0, len(df_box['ID'])))

    dist_2d_matrix = distance_2d(df_box, df_box, 'X', 'Y')
    dist_3d_matrix = distance_3d(df_box, df_box, 'X', 'Y', 'Elevation')
    print('3D distance matrix has been created')
    if np.any(dist_2d_matrix):

        edges_matrix = weight_matrix(df_box, dist_3d_matrix, line_bc)
        length_limit = resolution * 1.5
        edges_matrix[dist_2d_matrix > math.ceil(length_limit)] = 0
        edges_matrix_sparse = sparse.csr_matrix(edges_matrix)

        graph = nx.from_scipy_sparse_matrix(edges_matrix_sparse)
        source = df_box.loc[df_box['ID'] == int(assigned_substation['ID']), :]
        source = int(source.index.values)
        target = df_box.loc[df_box['ID'] == int(connecting_point['ID']), :]
        target = int(target.index.values)

        path = nx.dijkstra_path(graph, source, target, weight='weight')
        steps = len(path)

        # Creating the shapefile
        row = 0
        connection = gpd.GeoDataFrame(index=range(0, steps-1),
                                      columns=['ID1', 'ID2', 'Cost',
                                               'geometry'], crs=from_epsg(crs))

        for h in range(0, steps - 1):

            connection.at[row, 'geometry'] = LineString(
                [(df_box.loc[path[h], 'X'], df_box.loc[path[h], 'Y'],
                    df_box.loc[path[h], 'Elevation']),
                 (df_box.loc[path[h + 1], 'X'], df_box.loc[path[h + 1], 'Y'],
                    df_box.loc[path[h + 1], 'Elevation'])])

            # int here is necessary to use the command .to_file
            connection.at[row, 'ID1'] = int(df_box.loc[path[h], 'ID'])
            connection.at[row, 'ID2'] = int(df_box.loc[path[h + 1], 'ID'])
            connection.at[row, 'Cost'] = edges_matrix.loc[
                df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']]
            row += 1

        connection['Length'] = connection['geometry'].length
        connection_cost = int(connection['Cost'].sum(axis=0))
        connection_length = connection['Length'].sum(axis=0)

        return connection, connection_cost, connection_length
    elif not np.any(dist_2d_matrix):
        connection = gpd.GeoDataFrame()
        connection_cost = 0
        connection_length = 0
    return connection, connection_cost, connection_length


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
                direct_connection, direct_cost, direct_length = dijkstra_connection(
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

