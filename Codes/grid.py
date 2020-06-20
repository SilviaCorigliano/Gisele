import os
import pandas as pd
import numpy as np
import geopandas as gpd
from supporting_GISEle2 import *
from shapely.geometry import Point
from Codes import Steinerman, Spiderman, dijkstra


def routing(geo_df_clustered, geo_df, clusters_list, resolution,
            pop_threshold, input_sub, line_bc, limit_hv, limit_mv):
    s()
    print("3. Grid Creation")
    s()
    total_grid = total_connection = pd.DataFrame()
    grid_resume = pd.DataFrame(index=clusters_list.index,
                               columns=['Grid Length', 'Grid Cost',
                                        'Connection Length', 'Connection Cost',
                                        'Connection Type'])

    os.chdir(r'Input//')
    substations = pd.read_csv(input_sub + '.csv')
    geometry = [Point(xy) for xy in zip(substations['X'], substations['Y'])]
    substations = gpd.GeoDataFrame(substations, geometry=geometry,
                                   crs=geo_df.crs)
    os.chdir(r'..')
    os.chdir(r'Output//Grids')

    for cluster_n in clusters_list.Cluster:

        gdf_cluster = geo_df_clustered[
            geo_df_clustered['Cluster'] == cluster_n]
        gdf_cluster_pop = gdf_cluster[
            gdf_cluster['Population'] >= pop_threshold]

        points_to_electrify = int(len(gdf_cluster_pop))

        if points_to_electrify > 0:

            l()
            print("Creating grid for Cluster n." + str(cluster_n) + " of "
                  + str(len(clusters_list.Cluster)))
            l()

            c_grid, c_grid_cost, c_grid_length, c_grid_points = \
                cluster_grid(geo_df, gdf_cluster_pop, resolution,
                             line_bc, points_to_electrify)
            print("Cluster grid created")

            print('Assigning to the nearest substation.. ')
            assigned_substation, connecting_point, connection_type = \
                substation_assignment(cluster_n, geo_df, c_grid_points,
                                      substations, clusters_list, limit_hv,
                                      limit_mv)

            print('Connecting to the substation.. ')
            connection, connection_cost, connection_length = \
                dijkstra.dijkstra_connection(geo_df, assigned_substation,
                                             connecting_point, c_grid_points,
                                             line_bc, resolution)
            print("Substation connection created")

            c_grid.to_file('Grid_' + str(cluster_n) + '.shp')
            total_grid = gpd.GeoDataFrame(
                pd.concat([total_grid, c_grid], sort=True))

            if not connection.empty:
                connection.to_file('connection_' + str(cluster_n) + '.shp')
                total_connection = gpd.GeoDataFrame(pd.concat(
                    [total_connection, connection], sort=True))

            grid_resume.at[cluster_n, 'Connection Type'] = connection_type
            grid_resume.at[cluster_n, 'Grid Length'] = c_grid_length / 1000
            grid_resume.at[cluster_n, 'Grid Cost'] = c_grid_cost / 1000
            grid_resume.at[
                cluster_n, 'Connection Length'] = connection_length / 1000
            grid_resume.at[
                cluster_n, 'Connection Cost'] = connection_cost / 1000

        elif points_to_electrify == 0:

            grid_resume.at[cluster_n, 'Grid Length'] = 0
            grid_resume.at[cluster_n, 'Grid Cost'] = 0
            grid_resume.at[cluster_n, 'Connection Length'] = 0
            grid_resume.at[cluster_n, 'Connection Cost'] = 0

    grid_resume = clusters_list.join(grid_resume)
    grid_resume.to_csv('grid_resume.csv', index=False)
    total_grid.crs = total_connection.crs = geo_df.crs
    total_grid.to_file('all_cluster_grids')
    total_connection.to_file('all_connections')
    os.chdir(r'..//..')
    return grid_resume


def substation_assignment(cluster_n, geo_df, c_grid_points, substations,
                          clusters_list, limit_hv, limit_mv):

    if clusters_list.loc[cluster_n, 'Load'] > limit_hv:
        substations = substations[substations['Type'] == 'HV']
    elif limit_mv < clusters_list.loc[cluster_n, 'Load'] < limit_hv:
        substations = substations[substations['Type'] == 'MV']
    elif clusters_list.loc[cluster_n, 'Load'] < limit_mv:
        substations = substations[substations['Type'] != 'HV']

    substations = substations.assign(
        nearest_id=substations.apply(nearest, df=geo_df, src_column='ID',
                                     axis=1))

    sub_in_df = gpd.GeoDataFrame(crs=geo_df.crs)

    for i, row in substations.iterrows():
        sub_in_df = sub_in_df.append(
            geo_df[geo_df['ID'] == row['nearest_id']], sort=False)
    sub_in_df.reset_index(drop=True, inplace=True)

    grid_points = gpd.GeoDataFrame(crs=geo_df.crs)
    for i in np.unique(c_grid_points):
        grid_points = grid_points.append(
            geo_df[geo_df['ID'] == i], sort=False)

    dist = distance_3d(grid_points, sub_in_df, 'X', 'Y', 'Elevation')

    assigned_substation = sub_in_df[sub_in_df['ID'] == dist.min().idxmin()]
    connection_point = grid_points[grid_points['ID'] ==
                                   dist.min(axis=1).idxmin()]
    connection_type = substations[substations['nearest_id']
                                  == dist.min().idxmin()].Type.values[0]

    print('Substation assignment complete.')

    return assigned_substation, connection_point, connection_type


def cluster_grid(geo_df, gdf_cluster_pop, resolution, line_bc,
                 points_to_electrify):
    if points_to_electrify < 3:
        c_grid = gdf_cluster_pop
        c_grid_cost = 0
        c_grid_length = 0

    elif points_to_electrify < resolution:

        c_grid2, c_grid_cost2, c_grid_length2, c_grid_points2 = Spiderman. \
            spider(geo_df, gdf_cluster_pop, line_bc, resolution)

        c_grid1, c_grid_cost1, c_grid_length1, c_grid_points1 = Steinerman. \
            steiner(geo_df, gdf_cluster_pop, line_bc, resolution)

        if c_grid_cost1 <= c_grid_cost2:
            print("Steiner algorithm has the better cost")
            c_grid = c_grid1
            c_grid_length = c_grid_length1
            c_grid_cost = c_grid_cost1
            c_grid_points = c_grid_points1

        elif c_grid_cost1 > c_grid_cost2:
            print("Spider algorithm has the better cost")
            c_grid = c_grid2
            c_grid_length = c_grid_length2
            c_grid_cost = c_grid_cost2
            c_grid_points = c_grid_points2

    elif points_to_electrify >= resolution:
        print("Too many points to use Steiner, running Spider.")
        c_grid, c_grid_cost, c_grid_length, c_grid_points = Spiderman. \
            spider(geo_df, gdf_cluster_pop, line_bc, resolution)

    return c_grid, c_grid_cost, c_grid_length, c_grid_points


def connection_optimization(geo_df, grid_resume, resolution, line_bc, limit_hv,
                            limit_mv):
    os.chdir(r'Output//Grids')
    total_connections_opt = pd.DataFrame()
    check = pd.DataFrame(index=grid_resume.Cluster, columns=['Check'])
    check.loc[:, 'Check'] = False
    grid_resume = grid_resume.sort_values(by='Connection Cost')
    l()
    print('Optimization of substation connections')
    l()
    while not check.all().values:
        for i in grid_resume.Cluster:
            optimized = False
            grid_i = gpd.read_file("Grid_" + str(i) + ".shp")
            c_grid_points_i = list(zip(grid_i.ID1.astype(int),
                                       grid_i.ID2.astype(int)))
            grid_i = line_to_points(grid_i, geo_df)
            print('Evaluating if there is a better connection for cluster '
                  + str(i))

            exam = pd.DataFrame(index=grid_resume.index,
                                columns=['Distance', 'Cost'], dtype=int)
            if grid_resume.loc[i, 'Connection Cost'] == 0:
                check.loc[i, 'Check'] = True
                continue
            for j in grid_resume.Cluster:

                if i == j:
                    exam.Distance[j] = 9999999
                    exam.Cost[j] = 99999999
                    continue

                grid_j = gpd.read_file("Grid_" + str(j) + ".shp")
                c_grid_points = c_grid_points_i
                c_grid_points.append(list(zip(grid_j.ID1.astype(int),
                                              grid_j.ID2.astype(int))))
                grid_j = line_to_points(grid_j, geo_df)

                dist_2d = pd.DataFrame(distance_2d(grid_i, grid_j, 'X', 'Y'),
                                       index=grid_i.ID.values,
                                       columns=grid_j.ID.values)
                p1 = geo_df[geo_df['ID'] == dist_2d.min().idxmin()]
                p2 = geo_df[geo_df['ID'] == dist_2d.min(axis=1).idxmin()]

                # if the distance between clusters too high, skip
                if dist_2d.min().min() / 1000 > \
                        1.2 * (grid_resume.loc[i, 'Connection Length']):
                    exam.Distance[j] = 9999999
                    exam.Cost[j] = 99999999
                    continue

                connection, connection_cost, connection_length = \
                    dijkstra.dijkstra_connection(geo_df, p1, p2, c_grid_points,
                                                 line_bc, resolution)

                exam.Cost[j] = connection_cost
                exam.Distance[j] = connection_length

                if grid_resume.loc[j, 'Connection Type'] == 'MV' and \
                        grid_resume.loc[i, 'Load'] + \
                        grid_resume.loc[j, 'Load'] > limit_hv:
                    continue

                if grid_resume.loc[j, 'Connection Type'] == 'LV' and \
                        grid_resume.loc[i, 'Load'] + \
                        grid_resume.loc[j, 'Load'] > limit_mv:
                    continue

                if min(exam.Cost) == connection_cost and check.loc[j, 'Check']\
                    and connection_cost / 1000 < \
                        grid_resume.loc[i, 'Connection Cost']:
                    optimized = True
                    best_connection = connection

            if optimized:
                best_connection.to_file('NewConnection_' + str(i) + '.shp')
                grid_resume.loc[
                    i, 'Connection Length'] = min(exam.Distance) / 1000
                grid_resume.loc[i, 'Connection Cost'] = min(exam.Cost) / 1000
                grid_resume.loc[i, 'Connection Type'] = \
                    grid_resume.loc[exam['Cost'].idxmin(), 'Connection Type']
                print('A new connection for Cluster ' + str(
                    i) + ' was successfully created')
                total_connections_opt = gpd.GeoDataFrame(
                    pd.concat([total_connections_opt, best_connection],
                              sort=True))
            elif not optimized:
                print('No better connection for Cluster ' + str(
                    i) + '. Keeping the substation connection ')
                if grid_resume.loc[i, 'Connection Length'] > 0:
                    best_connection = gpd.read_file(
                        'connection_' + str(i) + '.shp')
                    total_connections_opt = gpd.GeoDataFrame(
                        pd.concat([total_connections_opt, best_connection],
                                  sort=True))

            check.loc[i, 'Check'] = True

    grid_resume = grid_resume.sort_values(by='Cluster')
    grid_resume.to_csv('grid_resume_opt.csv', index=False)
    total_connections_opt.crs = geo_df.crs
    total_connections_opt.to_file('all_connections_opt')
    print('All grids and connections have been optimized and exported.')
    os.chdir(r'..//..')
    return grid_resume
