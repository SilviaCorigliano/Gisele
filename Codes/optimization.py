import os
import pandas as pd
import geopandas as gpd
from supporting_GISEle2 import line_to_points, distance_2d, l
from Codes import dijkstra


def connections(geo_df, grid_resume, resolution, line_bc, step, input_sub):
    if step == 4:
        file = 'Branch_'
    else:
        file = 'Grid_'
    os.chdir('../..')

    substations = pd.read_csv(r'Input/' + input_sub + '.csv')
    os.chdir(r'Output/Grids')

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
            grid_i = gpd.read_file(file + str(i) + ".shp")
            c_grid_points_i = list(zip(grid_i.ID1.astype(int),
                                       grid_i.ID2.astype(int)))
            grid_i = line_to_points(grid_i, geo_df)
            print('Evaluating if there is a better connection for cluster '
                  + str(i))

            exam = pd.DataFrame(index=grid_resume.index,
                                columns=['Distance', 'Cost'], dtype=int)
            if grid_resume.loc[i, 'Connection Cost'] == 0:
                print('No better connection for Cluster ' + str(
                    i) + '. Keeping the substation connection ')
                check.loc[i, 'Check'] = True
                continue
            for j in grid_resume.Cluster:

                if i == j:
                    exam.Distance[j] = 9999999
                    exam.Cost[j] = 99999999
                    continue

                grid_j = gpd.read_file(file + str(j) + ".shp")
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
                        2 * (grid_resume.loc[i, 'Connection Length']):
                    exam.Distance[j] = 9999999
                    exam.Cost[j] = 99999999
                    continue

                connection, connection_cost, connection_length = \
                    dijkstra.dijkstra_connection(geo_df, p1, p2, c_grid_points,
                                                 line_bc, resolution)

                sub_id = grid_resume.loc[j, 'Substation ID']

                exam.Cost[j] = connection_cost
                exam.Distance[j] = connection_length

                if grid_resume.loc[i, 'Load'] + grid_resume.loc[j, 'Load'] \
                        > substations.loc[sub_id, 'PowerAvailable']:
                    continue

                if min(exam.Cost) == connection_cost and check.loc[j, 'Check']\
                    and connection_cost / 1000 < \
                        grid_resume.loc[i, 'Connection Cost']:
                    optimized = True
                    best_connection = connection

            if optimized:
                best_connection.to_file('Connection_' + str(i) + '.shp')
                grid_resume.loc[
                    i, 'Connection Length'] = min(exam.Distance) / 1000
                grid_resume.loc[i, 'Connection Cost'] = min(exam.Cost) / 1000
                grid_resume.loc[i, 'Connection Type'] = \
                    grid_resume.loc[exam['Cost'].idxmin(), 'Connection Type']
                grid_resume.loc[i, 'Substation ID'] = \
                    grid_resume.loc[exam['Cost'].idxmin(), 'Substation ID']
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
