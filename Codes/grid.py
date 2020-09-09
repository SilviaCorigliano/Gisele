import os
import pandas as pd
import numpy as np
import geopandas as gpd
from supporting_GISEle2 import *
from shapely.geometry import Point
from Codes import Steinerman, Spiderman, dijkstra


def routing(geo_df_clustered, geo_df, clusters_list, resolution,
            pop_thresh, input_sub, line_bc, sub_cost_hv, sub_cost_mv):
    s()
    print("3. Grid Creation")
    s()
    total_grid = total_connection = pd.DataFrame()
    grid_resume = pd.DataFrame(index=clusters_list.index,
                               columns=['Grid Length [km]', 'Grid Cost [k€]',
                                        'Connection Length [km]',
                                        'Connection Cost [k€]',
                                        'Connection Type', 'Substation ID'])

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
            gdf_cluster['Population'] >= pop_thresh]

        n_terminal_nodes = int(len(gdf_cluster_pop))

        if n_terminal_nodes > 1:

            l()
            print("Creating grid for Cluster n." + str(cluster_n) + " of "
                  + str(len(clusters_list.Cluster)))
            l()

            c_grid, c_grid_cost, c_grid_length, c_grid_points = \
                cluster_grid(geo_df, gdf_cluster_pop, resolution,
                             line_bc, n_terminal_nodes, gdf_cluster)
            print("Cluster grid created")

            print('Assigning to the nearest 5 substations.. ')
            assigned_substations, connecting_points = \
                substation_assignment(cluster_n, geo_df, c_grid_points,
                                      substations, clusters_list)

            print('Connecting to the substation.. ')

            connection, connection_cost, connection_length, connection_type, \
                connection_id = substation_connection(geo_df, substations,
                                                      assigned_substations,
                                                      connecting_points,
                                                      c_grid_points,
                                                      line_bc, resolution,
                                                      sub_cost_hv, sub_cost_mv)

            print("Substation connection created")

            c_grid.to_file('Grid_' + str(cluster_n) + '.shp')
            total_grid = gpd.GeoDataFrame(
                pd.concat([total_grid, c_grid], sort=True))

            if not connection.empty:
                connection.to_file('Connection_' + str(cluster_n) + '.shp')
                total_connection = gpd.GeoDataFrame(pd.concat(
                    [total_connection, connection], sort=True))

            grid_resume.loc[cluster_n, 'Connection Type'] = connection_type
            grid_resume.loc[cluster_n, 'Substation ID'] = connection_id
            grid_resume.loc[cluster_n, 'Grid Length [km]'] = c_grid_length / 1000
            grid_resume.loc[cluster_n, 'Grid Cost [k€]'] = c_grid_cost / 1000
            grid_resume.loc[
                cluster_n, 'Connection Length [km]'] = connection_length / 1000
            grid_resume.loc[
                cluster_n, 'Connection Cost [k€]'] = connection_cost / 1000

        elif n_terminal_nodes == 0:

            grid_resume.at[cluster_n, 'Grid Length [km]'] = 0
            grid_resume.at[cluster_n, 'Grid Cost [k€]'] = 0
            grid_resume.at[cluster_n, 'Connection Length [km]'] = 0
            grid_resume.at[cluster_n, 'Connection Cost [k€]'] = 0

    grid_resume = clusters_list.join(grid_resume)
    grid_resume.to_csv('grid_resume.csv', index=False)
    total_grid.crs = total_connection.crs = geo_df.crs
    total_grid.to_file('all_cluster_grids')
    total_connection.to_file('all_connections')
    os.chdir(r'../..')
    return grid_resume, substations


def substation_assignment(cluster_n, geo_df, c_grid_points, substations,
                          clusters_list):

    substations = substations[substations['PowerAvailable']
                              > clusters_list.loc[cluster_n, 'Load [kW]']]
    substations = substations.assign(
        nearest_id=substations.apply(nearest, df=geo_df, src_column='ID',
                                     axis=1))

    sub_in_df = gpd.GeoDataFrame(crs=geo_df.crs, geometry=[])

    for i, row in substations.iterrows():
        sub_in_df = sub_in_df.append(
            geo_df[geo_df['ID'] == row['nearest_id']], sort=False)
    sub_in_df.reset_index(drop=True, inplace=True)

    grid_points = gpd.GeoDataFrame(crs=geo_df.crs, geometry=[])
    for i in np.unique(c_grid_points):
        grid_points = grid_points.append(
            geo_df[geo_df['ID'] == i], sort=False)

    dist = distance_3d(grid_points, sub_in_df, 'X', 'Y', 'Elevation')

    assigned_substations = dist.min().nsmallest(3).index.values

    connecting_points = dist.idxmin()
    connecting_points = connecting_points[
        connecting_points.index.isin(assigned_substations)].values

    assigned_substations = \
        sub_in_df[sub_in_df['ID'].isin(assigned_substations)]
    sub_ids = substations[substations['nearest_id'].isin(
        assigned_substations['ID'])].index.values
    assigned_substations = \
        assigned_substations.assign(sub_id=sub_ids)
    assigned_substations.reset_index(drop=True, inplace=True)

    print('Substation assignment complete')

    return assigned_substations, connecting_points


def cluster_grid(geo_df, gdf_cluster_pop, resolution, line_bc,
                 n_terminal_nodes, gdf_cluster):

    c_grid = gdf_cluster_pop
    c_grid_cost = 0
    c_grid_length = 0
    c_grid_points = []

    if n_terminal_nodes < resolution/5 and gdf_cluster.length.size < 500:

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

    else:
        print("Too many points to use Steiner, running Spider.")
        c_grid, c_grid_cost, c_grid_length, c_grid_points = Spiderman. \
            spider(geo_df, gdf_cluster_pop, line_bc, resolution)

    return c_grid, c_grid_cost, c_grid_length, c_grid_points


def substation_connection(geo_df, substations, assigned_substations,
                          connecting_point, c_grid_points, line_bc,
                          resolution, sub_cost_hv, sub_cost_mv):

    best_connection = pd.DataFrame()
    best_connection_cost = 0
    best_connection_length = 0
    best_connection_type = []
    best_connection_power = 0
    connection_id = []

    costs = np.full((1, len(assigned_substations)), 9999999)[0]
    for row in assigned_substations.iterrows():
        point = geo_df[geo_df['ID'] == connecting_point[row[0]]]
        sub = geo_df[geo_df['ID'] == row[1].ID]

        connection, connection_cost, connection_length = \
            dijkstra.dijkstra_connection(geo_df, sub, point, c_grid_points,
                                         line_bc, resolution)

        if substations.loc[row[1]['sub_id'], 'Exist'] == 'no':
            if substations.loc[row[1]['sub_id'], 'Type'] == 'HV':
                connection_cost = connection_cost + sub_cost_hv
            elif substations.loc[row[1]['sub_id'], 'Type'] == 'MV':
                connection_cost = connection_cost + sub_cost_mv
        costs[row[0]] = connection_cost
        connection_power = substations.loc[row[1]['sub_id'], 'PowerAvailable']
        if min(costs) == connection_cost:
            if (best_connection_cost - connection_cost) < 5000 \
                    and (best_connection_power - connection_power) > 100:
                continue
            best_connection = connection
            best_connection_cost = connection_cost
            best_connection_length = connection_length
            best_connection_type = substations.loc[row[1]['sub_id'], 'Type']
            best_connection_power = substations.loc[row[1]['sub_id'],
                                                    'PowerAvailable']
            connection_id = row[1]['sub_id']
    return best_connection, best_connection_cost, best_connection_length, \
        best_connection_type, connection_id
