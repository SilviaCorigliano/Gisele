"""
Main branches and collateral algorithm

Created on Mon Jan 27 2020

@author: Vinicius Gadelha
"""

import os
import networkx as nx
from shapely.ops import nearest_points
from Codes.remove_cycles_weight_final import *
from shapely.geometry import Point
from Codes.grid import*
from Codes import initialization, clustering, grid, dijkstra, Steinerman

os.chdir(r'..//')
step = 3
df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv, geo_df_clustered, \
        clusters_list, = initialization.import_csv_file(step)

geo_df, pop_points = initialization. \
    creating_geodataframe(df_weighted, crs, unit, input_csv, step)

print("Clusters successfully imported")

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

grid_resume = pd.DataFrame(index=clusters_list.Cluster,
                           columns=['Cluster', 'Chunk_Length', 'Chunk_Cost',
                                    'Grid_Length', 'Grid_Cost',
                                    'Connection_Length', 'Connection_Cost',
                                    'ConnectionType', 'Link_Length',
                                    'Link_Cost'])

grid_resume['Cluster'] = clusters_list['Cluster']
grid_resume['ClusterLoad'] = clusters_list['Load']
all_branch = pd.DataFrame()
all_collateral = pd.DataFrame()
all_connections = pd.DataFrame()
all_link = pd.DataFrame()



pop_thresh = 30
# ---------------------START OF THE MAIN CHUNK CREATION-----------------------

for i in clusters_list.Cluster:

    gdf_cluster_only = geo_df_clustered[geo_df_clustered['Cluster'] == i]
    gdf_zoomed_pop = gdf_zoomed[gdf_zoomed['Cluster'] == i]

    gdf_zoomed_pop = gdf_zoomed_pop[gdf_zoomed_pop['Population'] >= pop_thresh]

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

all_branch.crs = geo_df.crs
all_branch.to_file('all_branch')

#  -------------------START OF THE COLLATERAL TECHNIQUES----------------------

pop_thresh = 1
line_bc = 3333
for i in clusters_list.Cluster:

    gdf_clusters_pop = geo_df_clustered[geo_df_clustered['Cluster'] == i]
    gdf_clusters_pop = gdf_clusters_pop[gdf_clusters_pop
                                        ['Population'] >= pop_thresh]
    l()
    print('Creating collaterals for Cluster ' + str(i))
    l()
    if grid_resume.loc[i, 'Chunk_Length'] == 0:

        collateral, collateral_cost, collateral_length, collateral_points = \
            Steinerman.steiner(geo_df, gdf_clusters_pop, line_bc, resolution)

        print('Assigning to the nearest substation.. ')
        assigned_substation, connecting_point, connection_type = \
            substation_assignment(i, geo_df, collateral_points,
                                  substations, clusters_list, limit_hv,
                                  limit_mv)
        print('Connecting to the substation.. ')
        connection, connection_cost, connection_length = \
            dijkstra.dijkstra_connection(geo_df, assigned_substation,
                                         connecting_point, collateral_points,
                                         line_bc, resolution)
        print("Substation connection created")

        collateral.to_file("Branch_" + str(i) + '.shp')
        grid_resume.loc[i, 'Grid_Length'] = collateral_length / 1000
        grid_resume.loc[i, 'Grid_Cost'] = collateral_cost / 1000
        all_collateral = gpd.GeoDataFrame(pd.concat([all_collateral,
                                                     collateral], sort=True))
        if connection_cost > 100:
            all_connections = gpd.GeoDataFrame(
                pd.concat([all_connections, connection], sort=True))
            connection.to_file("Connection_" + str(i) + '.shp')
        grid_resume.loc[i, 'Connection_Length'] = connection_length / 1000
        grid_resume.loc[i, 'Connection_Cost'] = connection_cost / 1000
        continue

    # ----- ASSIGNING WEIGHT EQUAL TO 0 TO THE MAIN BRANCH -------------------

    chunk = gpd.read_file("Branch_" + str(i) + ".shp")
    branch_points = list(zip(chunk.ID1.astype(int),
                             chunk.ID2.astype(int)))
    nodes = list(chunk.ID1.astype(int)) + list(chunk.ID2.astype(int))
    nodes = list(dict.fromkeys(nodes))

    for n in nodes:
        geo_df.loc[geo_df['ID'] == n, 'Weight'] = 0.001

    collateral, collateral_cost, collateral_length, collateral_points = \
        Steinerman.steiner(geo_df, gdf_clusters_pop, line_bc, resolution)

    # for value in collateral.values:
    #     if value[2] in chunk.values and value[3] in chunk.values:
    #         collateral = collateral.drop((np.where(collateral.values[:, 1]
    #         == value[1])[0][0]))
    #         collateral = collateral.reset_index(drop=True)

    # collateral_cost = sum(collateral['Cost'])

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

    grid_resume.loc[i, 'Grid_Cost'] = collateral_cost / 1000
    grid_resume.loc[i, 'Grid_Length'] = collateral_length / 1000
    collateral.to_file("Collateral_" + str(i) + '.shp')
    all_collateral = gpd.GeoDataFrame(pd.concat([all_collateral, collateral],
                                      sort=True))

all_collateral.crs = all_connections.crs = geo_df.crs
all_collateral.to_file('all_collateral')
all_connections.to_file('all_connections')
#



# all_collateral = gpd.read_file('all_collateral.shp')

#  connecting points outside of clusters
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

    link, link_cost, link_length = dijkstra.dijkstra_connection(geo_df, p1, p2,
                                                                grid_points,
                                                                line_bc,
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
    unique_line = all_link[all_link.index == unique_line.first_valid_index()]
    all_link_no_dup = gpd.GeoDataFrame(pd.concat([all_link_no_dup,
                                                  unique_line], sort=True))

grid_resume.loc[0, 'Link_Length'] = all_link_no_dup.Length.sum() / 1000
grid_resume.loc[0, 'Link_Cost'] = all_link_no_dup.Cost.sum() / 1000
all_link_no_dup.crs = geo_df.crs
all_link_no_dup.to_file('all_link')
grid_resume.to_csv('grid_resume.csv', index=False)

print('100% electrification achieved')
# grid_optimized = grid_optimization(geo_df_clustered, geodf_in, grid_resume, crs, resolution, line_bc)


# --------- SIZING EACH DERIVATION ------------------------------------
for cluster in clusters_list:
    if grid_resume.at[cluster, 'Chunk_Length'] == 0:
        continue
    derivations = gpd.read_file('Steiner_Grid' + str(cluster) + '.shp')
    chunk = gpd.read_file('Chunk_' + str(cluster) + '.shp')
    for i in derivations.values:
        if i[2] in chunk.values and i[3] in chunk.values:
            derivations = derivations.drop((np.where(derivations.values[:, 1] == i[1])[0][0]))
            derivations = derivations.reset_index(drop=True)
    count = 1
    G = nx.Graph()
    G.add_edges_from([*zip(list(derivations.ID1),list(derivations.ID2))])
    components = list(nx.algorithms.components.connected_components(G))
    for i in components:
        points = np.array(list(i))
        pop_derivation = 0
        for j in points:
            for k in range(derivations.values.__len__()):
                if derivations.at[k,'ID1'] == j or derivations.at[k,'ID2'] == j:
                    derivations.at[k, 'id'] = count
            pop_derivation = pop_derivation + geo_df_clustered[geo_df_clustered['ID'] == j].Population.values
        derivations.loc[derivations['id'] == count, 'Power'] = int(pop_derivation)*0.4
        count = count+1
    all_links = gpd.GeoDataFrame(pd.concat([all_links, derivations], sort=True))
all_links.crs = geo_df.crs
all_links.to_file('derivations')



