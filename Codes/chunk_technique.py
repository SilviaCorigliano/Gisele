import os
import networkx as nx
from shapely.ops import nearest_points
from Codes.remove_cycles_weight_final import *
from shapely.geometry import Point
from Codes.grid import *
from Codes import initialization, clustering, grid, dijkstra, Steinerman


def collateral_sizing(geo_df_clustered, col, pop_load):

    collateral_sized = pd.DataFrame()
    col.drop(col[col['Cost'] < 100].index, inplace=True)
    col.reset_index(drop=True, inplace=True)
    count = 1
    graph = nx.Graph()
    graph.add_edges_from([*zip(list(col.ID1), list(col.ID2))])

    components = list(nx.algorithms.components.connected_components(graph))

    for i in components:
        points = np.array(list(i))
        pop_col = 0
        for j in points.astype(int):
            col.loc[col['ID1'] == j, 'id'] = count
            col.loc[col['ID2'] == j, 'id'] = count
            pop_col = pop_col + geo_df_clustered[geo_df_clustered
                                                 ['ID'] == j].Population
        col.loc[col['id'] == count, 'Power'] = int(pop_col) * pop_load
        count += 1

    collateral_sized = gpd.GeoDataFrame(pd.concat([collateral_sized,
                                                   col], sort=True))
    collateral_sized.crs = geo_df_clustered.crs
    collateral_sized.to_file('derivations')

    return collateral_sized


def routing(geo_df_clustered, geo_df, clusters_list, resolution,
            pop_thresh, pop_thresh_lr, input_sub, line_bc, limit_hv, limit_mv):

    grid_resume = pd.DataFrame(index=clusters_list.Cluster,
                               columns=['Cluster', 'Chunk_Length',
                                        'Chunk_Cost',
                                        'Grid_Length', 'Grid_Cost',
                                        'Connection_Length', 'Connection_Cost',
                                        'ConnectionType', 'Link_Length',
                                        'Link_Cost'])

    grid_resume['Cluster'] = clusters_list['Cluster']
    grid_resume['ClusterLoad'] = clusters_list['Load']

    #  IMPORTING THE LOWER RESOLUTION GRID
    os.chdir(r'Input//')
    zoom = pd.read_csv('Cavalcante-Brazil-32723-4000.csv')
    geometry = [Point(xy) for xy in zip(zoom['X'], zoom['Y'])]
    gdf_zoomed = gpd.GeoDataFrame(zoom, geometry=geometry, crs=geo_df.crs)

    #  IMPORTING SUBSTATIONS

    substations = pd.read_csv(input_sub + '.csv')
    geometry = [Point(xy) for xy in zip(substations['X'], substations['Y'])]
    substations = gpd.GeoDataFrame(substations, geometry=geometry,
                                   crs=geo_df.crs)
    os.chdir(r'..')
    os.chdir(r'Output//Main Branch')

    grid_resume = main_branch(gdf_zoomed, geo_df_clustered, clusters_list,
                              resolution, pop_thresh_lr, line_bc, grid_resume)

    line_bc = line_bc/3
    grid_resume, all_collateral = collateral(geo_df_clustered, geo_df,
                                             clusters_list, substations,
                                             resolution, pop_thresh, line_bc,
                                             limit_hv, limit_mv, grid_resume)

    links(geo_df_clustered, geo_df, all_collateral, resolution, line_bc,
          grid_resume)

    return grid_resume


def main_branch(gdf_zoomed, geo_df_clustered, clusters_list, resolution,
                pop_thresh_lr, line_bc, grid_resume):
    all_branch = pd.DataFrame()
    for i in clusters_list.Cluster:

        gdf_cluster_only = geo_df_clustered[geo_df_clustered['Cluster'] == i]
        gdf_zoomed_pop = gdf_zoomed[gdf_zoomed['Cluster'] == i]

        gdf_zoomed_pop = gdf_zoomed_pop[gdf_zoomed_pop['Population'] >=
                                        pop_thresh_lr]

        points_to_electrify = int(len(gdf_zoomed_pop))
        l()
        print("Creating main branch for Cluster n." + str(i) + " of "
              + str(len(clusters_list.Cluster)))
        l()
        if points_to_electrify > 1:

            branch, branch_cost, branch_length, branch_points = Steinerman. \
                steiner(gdf_cluster_only, gdf_zoomed_pop, line_bc, resolution)

            print("Cluster main branch created")

            grid_resume.at[i, 'Chunk_Length'] = branch_length / 1000
            grid_resume.at[i, 'Chunk_Cost'] = branch_cost / 1000
            branch.to_file("Branch_" + str(i) + '.shp')
            all_branch = gpd.GeoDataFrame(pd.concat([all_branch, branch],
                                                    sort=True))
        else:
            print('Main branch for Cluster ' + str(i) + ' not necessary')
            grid_resume.at[i, 'Chunk_Length'] = 0
            grid_resume.at[i, 'Chunk_Cost'] = 0

    all_branch.crs = geo_df_clustered.crs
    all_branch.to_file('all_branch')

    return grid_resume
#  -------------------START OF THE COLLATERAL TECHNIQUES----------------------


def collateral(geo_df_clustered, geo_df, clusters_list, substations,
               resolution, pop_thresh, line_bc, limit_hv, limit_mv,
               grid_resume):

    all_connections = pd.DataFrame()
    all_collateral = pd.DataFrame()
    for i in clusters_list.Cluster:

        gdf_clusters_pop = geo_df_clustered[geo_df_clustered['Cluster'] == i]
        gdf_clusters_pop = gdf_clusters_pop[gdf_clusters_pop
                                            ['Population'] >= pop_thresh]
        l()
        print("Creating collateral's for Cluster " + str(i))
        l()
        if grid_resume.loc[i, 'Chunk_Length'] == 0:

            col, col_cost, col_length, col_points = \
                Steinerman.steiner(geo_df, gdf_clusters_pop, line_bc,
                                   resolution)

            print('Assigning to the nearest substation.. ')
            assigned_substation, connecting_point, connection_type = \
                substation_assignment(i, geo_df, col_points,
                                      substations, clusters_list, limit_hv,
                                      limit_mv)
            print('Connecting to the substation.. ')
            connection, connection_cost, connection_length = \
                dijkstra.dijkstra_connection(geo_df, assigned_substation,
                                             connecting_point, col_points,
                                             line_bc, resolution)
            print("Substation connection created")

            col.to_file("Branch_" + str(i) + '.shp')
            grid_resume.loc[i, 'Grid_Length'] = col_length / 1000
            grid_resume.loc[i, 'Grid_Cost'] = col_cost / 1000
            all_collateral = gpd.GeoDataFrame(pd.concat([all_collateral,
                                                         col], sort=True))
            if connection_cost > 100:
                all_connections = gpd.GeoDataFrame(
                    pd.concat([all_connections, connection], sort=True))
                connection.to_file("Connection_" + str(i) + '.shp')
            grid_resume.loc[i, 'Connection_Length'] = connection_length / 1000
            grid_resume.loc[i, 'Connection_Cost'] = connection_cost / 1000

            continue

        # ----- ASSIGNING WEIGHT EQUAL TO 0 TO THE MAIN BRANCH ---------------

        branch = gpd.read_file("Branch_" + str(i) + ".shp")
        branch_points = list(zip(branch.ID1.astype(int),
                                 branch.ID2.astype(int)))
        nodes = list(branch.ID1.astype(int)) + list(branch.ID2.astype(int))
        nodes = list(dict.fromkeys(nodes))

        for n in nodes:
            geo_df.loc[geo_df['ID'] == n, 'Weight'] = 0.0001

        col, col_cost, col_length, col_points = \
            Steinerman.steiner(geo_df, gdf_clusters_pop, line_bc, resolution)

        col, col_cost, col_length = collateral_sizing(col, branch, col_points)

        print('Assigning to the nearest substation.. ')
        assigned_substation, connecting_point, connection_type = \
            substation_assignment(i, geo_df, branch_points,
                                  substations, clusters_list, limit_hv,
                                  limit_mv)

        print('Connecting to the substation.. ')
        connection, connection_cost, connection_length = \
            dijkstra.dijkstra_connection(geo_df, assigned_substation,
                                         connecting_point, branch_points,
                                         line_bc, resolution)
        print("Substation connection created")

        if connection_cost > 100:
            all_connections = gpd.GeoDataFrame(
                pd.concat([all_connections, connection], sort=True))
            connection.to_file("Connection_" + str(i) + '.shp')
        grid_resume.loc[i, 'Connection_Length'] = connection_length / 1000
        grid_resume.loc[i, 'Connection_Cost'] = connection_cost / 1000

        grid_resume.loc[i, 'Grid_Cost'] = col_cost / 1000
        grid_resume.loc[i, 'Grid_Length'] = col_length / 1000
        col.to_file("Collateral_" + str(i) + '.shp')
        all_collateral = gpd.GeoDataFrame(pd.concat([all_collateral, col],
                                                    sort=True))

    all_collateral.crs = all_connections.crs = geo_df.crs
    all_collateral.to_file('all_collateral')
    all_connections.to_file('all_connections')
    return grid_resume


def links(geo_df_clustered, geo_df, all_collateral, resolution, line_bc,
          grid_resume):

    all_link = pd.DataFrame()
    l()
    print('Connecting all the people outside the clustered area..')
    l()

    # all_collateral.reset_index(inplace=True, drop=True)

    gdf_clusters_out = geo_df_clustered[geo_df_clustered['Cluster'] == -1]
    gdf_clusters_out = gdf_clusters_out[gdf_clusters_out['Population'] >= 1]

    grid_points = list(zip(all_collateral.ID1.astype(int),
                           all_collateral.ID2.astype(int)))
    all_points = line_to_points(all_collateral, geo_df)
    gdf_clusters_out = gdf_clusters_out.assign(
        nearest_id=gdf_clusters_out.apply(nearest, df=all_points,
                                          src_column='ID', axis=1))

    for row in gdf_clusters_out.iterrows():
        p1 = geo_df_clustered[geo_df_clustered['ID'] == row[1].ID]
        p2 = geo_df_clustered[geo_df_clustered['ID'] == row[1].nearest_id]

        link, link_cost, link_length = \
            dijkstra.dijkstra_connection(geo_df, p1, p2, grid_points, line_bc,
                                         resolution)

        all_link = gpd.GeoDataFrame(pd.concat([all_link, link], sort=True))

    #  REMOVING DUPLICATIONS
    nodes = list(zip(all_link.ID1.astype(int), all_link.ID2.astype(int)))
    nodes = list(set(nodes))
    all_link.reset_index(drop=True, inplace=True)
    all_link_no_dup = pd.DataFrame()

    for pair in nodes:
        unique_line = all_link.loc[(all_link['ID1'] == pair[0]) & (
                all_link['ID2'] == pair[1])]
        unique_line = all_link[all_link.index ==
                               unique_line.first_valid_index()]
        all_link_no_dup = gpd.GeoDataFrame(pd.concat([all_link_no_dup,
                                                      unique_line], sort=True))

    grid_resume.loc[0, 'Link_Length'] = all_link_no_dup.Length.sum() / 1000
    grid_resume.loc[0, 'Link_Cost'] = all_link_no_dup.Cost.sum() / 1000
    all_link_no_dup.crs = geo_df.crs
    all_link_no_dup.to_file('all_link')
    grid_resume.to_csv('grid_resume.csv', index=False)

    print('100% electrification achieved')

    return grid_resume
#  OPTIMIZE THE GRID
