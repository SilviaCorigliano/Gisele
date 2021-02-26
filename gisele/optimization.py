import os
import itertools
import pandas as pd
import geopandas as gpd
import numpy as np
from functions import line_to_points, distance_2d, l, nearest
from gisele import dijkstra, lcoe_optimization
from gisele.emission import emission_factor


def clusters_interconnections(geo_df_clustered, grid_resume, substations, mg, total_energy,
              grid_om, coe, grid_ir, grid_lifetime, branch, line_bc,
              resolution):
    substations = substations.assign(
        nearest_id=substations.apply(nearest, df=geo_df_clustered,
                                     src_column='ID', axis=1))
    gdf_roads = gpd.read_file('Output/Datasets/Roads/gdf_roads.shp')
    roads_segments =gpd.read_file('Output/Datasets/Roads/roads_segments.shp')

    if branch == 'yes':
        file = 'Branch_'
        os.chdir(r'Output/Branches')
    else:
        file = 'Grid_'
        os.chdir(r'Output/Grids')

    milp_clusters = grid_resume[['Cluster', 'Load [kW]']].copy()
    milp_clusters['Cluster'] = ['C' + str(i[0]) for i in
                                milp_clusters['Cluster'].iteritems()]
    # energy_mismatch = \
    #     (total_energy['Energy'] / 1000) / mg['Energy Produced [MWh]']
    milp_clusters['mg_npc'] = mg['Total Cost [k€]']
    milp_subs = substations[['ID', 'PowerAvailable']].copy()
    milp_subs['ID'] = ['S' + str(i[1]) for i in milp_subs['ID'].iteritems()]
    milp_subs['subs_npc'] = 10 #da cambiare
    sets = milp_clusters['Cluster'].append(milp_subs['ID'], ignore_index=True)
    combinations = list(itertools.combinations(sets, 2))
    milp_links = pd.DataFrame(index=range(combinations.__len__()),
                              columns=['0', '1'])
    milp_links['0'] = [i[0] for i in combinations]
    milp_links['1'] = [i[1] for i in combinations]
    milp_links['Cost'] = 999999
    for row in milp_links.iterrows():
        if 'S' in row[1][0] and 'S' in row[1][1]:
            continue
        c_grid_points = []
        print('Connecting ' + row[1][0] + ' and ' + row[1][1])
        if 'C' in row[1][0]:
            if os.path.isfile(file + str(row[1][0].split('C')[1]) +
                                   ".shp"):
                grid_1 = gpd.read_file(file + str(row[1][0].split('C')[1]) +
                                       ".shp")
                c_grid_points = list(zip(grid_1.ID1.astype(int),
                                         grid_1.ID2.astype(int)))
                grid_1 = line_to_points(grid_1, geo_df_clustered)
            else:
                grid_1=geo_df_clustered[geo_df_clustered['Cluster']==int(row[1][0].split('C')[1])]
                grid_1=grid_1[grid_1['Population']>0]
                c_grid_points = []
        elif 'S' in row[1][0]:
            sub_in_df = substations[
                substations['ID'] ==
                int(row[1][0].split('S')[1])].nearest_id.values[0]
            grid_1 = geo_df_clustered[geo_df_clustered['ID'] == sub_in_df]
        if 'C' in row[1][1]:
            if os.path.isfile(file + str(row[1][1].split('C')[1]) +
                              ".shp"):
                grid_2 = gpd.read_file(file + str(row[1][1].split('C')[1]) +
                                   ".shp")
                c_grid_points.append(list(zip(grid_2.ID1.astype(int),
                                          grid_2.ID2.astype(int))))
                grid_2 = line_to_points(grid_2, geo_df_clustered)
            else:
                grid_2 = geo_df_clustered[
                    geo_df_clustered['Cluster'] == int(row[1][1].split('C')[1])]
                grid_2 = grid_2[grid_2['Population'] > 0]

        elif 'S' in row[1][1]:
            sub_in_df = substations[
                substations['ID'] ==
                int(row[1][1].split('S')[1])].nearest_id.values[0]
            grid_2 = geo_df_clustered[geo_df_clustered['ID'] == sub_in_df]

        dist_2d = pd.DataFrame(distance_2d(grid_1, grid_2, 'X', 'Y'),
                               index=grid_1.ID.values,
                               columns=grid_2.ID.values)

        p1 = geo_df_clustered[geo_df_clustered['ID'] == dist_2d.min().idxmin()]
        p2 = geo_df_clustered[geo_df_clustered['ID'] ==
                              dist_2d.min(axis=1).idxmin()]

        # connection, connection_cost, connection_length, _ = \
            # dijkstra.dijkstra_connection(geo_df_clustered, p1, p2,
            #                              c_grid_points, line_bc, resolution)



        connection, connection_cost, connection_length, _ = \
            dijkstra.dijkstra_connection_roads(geo_df_clustered, p1, p2,
                                               c_grid_points, line_bc,
                                               resolution, gdf_roads,
                                               roads_segments)

        if connection.empty and connection_cost == 999999:
            continue
        elif connection.empty:
            connection_cost = 1000
            connection_length = 1000
        connection_om = [(connection_cost/1000) * grid_om] * 20
        connection_om = np.npv(grid_ir, connection_om)
        connection_salvage = (connection_cost/1000) * \
                             (grid_lifetime-20)/(grid_lifetime) #controllare costo
        milp_links.loc[row[0], 'Cost'] = (connection_cost / 1000) \
            + connection_om - connection_salvage
    milp_links.drop(milp_links[milp_links['Cost'] == 999999].index,
                    inplace=True)
    milp_links.reset_index(inplace=True, drop=True)
    #includo un valore fittizio per le emissioni legate alla costruzione dei link:
    #todo -> aggiornare con valore sensato di emissioni per combustibile prese da letteratura

    em_links = milp_links.copy()
    em_links['emission'] =(em_links['Cost']-min(em_links['Cost']))/(max(em_links['Cost'])-min(em_links['Cost']))*10
    em_links.drop('Cost',axis=1,inplace=True)

    os.chdir('../..')
    milp_links.to_csv(r'Output/LCOE/milp_links.csv', index=False)
    # aggiunto un valore di emissione legato alla lunghezza dei link
    em_links.to_csv(r'Output/LCOE/em_links.csv', index=False)
    sets.to_csv(r'Output/LCOE/set.csv', index=False)
    milp_links.loc[:, ['0', '1']].to_csv(r'Output/LCOE/possible_links.csv',
                                         index=False)

    milp_subs.loc[:, ['ID', 'PowerAvailable']].to_csv(
        r'Output/LCOE/sub_power.csv', index=False)

    milp_subs.loc[:, ['ID', 'subs_npc']].to_csv(r'Output/LCOE/sub_npc.csv',
                                                index=False)

    milp_subs.loc[:, 'ID'].to_csv(r'Output/LCOE/subs.csv', index=False)

    milp_clusters.loc[:, ['Cluster']].to_csv(r'Output/LCOE/clusters.csv',
                                             index=False)

    milp_clusters.loc[:, ['Cluster', 'Load [kW]']].to_csv(
        r'Output/LCOE/c_power.csv', index=False)

    milp_clusters.loc[:, ['Cluster', 'mg_npc']].to_csv(
        r'Output/LCOE/c_npc.csv', index=False)

    # total_energy['Cluster'] = ['C' + str(i[0]) for i in
    #                            total_energy['Cluster'].iteritems()]
    # total_energy.to_csv(r'Output/LCOE/energy.csv', index=False)

    mg_energy=mg[['Cluster','Energy Demand [MWh]']]
    mg_energy['Cluster'] = ['C' + str(i[0]) for i in
                               mg_energy['Cluster'].iteritems()]
    mg_energy.to_csv(r'Output/LCOE/energy.csv', index=False)

    mg_emissions=mg[['Cluster','CO2 [kg]']]
    mg_emissions['Cluster'] = ['C' + str(i[0]) for i in
                               mg_emissions['Cluster'].iteritems()]
    mg_emissions.to_csv(r'Output/LCOE/emissions.csv', index=False)

    return

def milp_execution(geo_df_clustered, grid_resume, substations, coe, branch, line_bc,
              resolution,p_max_lines):
    total_connections_opt = pd.DataFrame()

    # run the effective optimization
    nation_emis = 1000  # kgCO2 emission per kWh given country energy mix
    country = 'Mozambique'
    nation_emis = emission_factor(country)  # kgCO2/kWh

    lcoe_optimization.cost_optimization(p_max_lines, coe, nation_emis)

    gdf_roads = gpd.read_file('Output/Datasets/Roads/gdf_roads.shp')
    roads_segments = gpd.read_file('Output/Datasets/Roads/roads_segments.shp')
    con_out = pd.read_csv(r'Output/LCOE/MV_connections_output.csv')
    mg_out = pd.read_csv(r'Output/LCOE/MV_SHS_output.csv')

    substations = substations.assign(
        nearest_id=substations.apply(nearest, df=geo_df_clustered,
                                     src_column='ID', axis=1))

    dups = con_out.duplicated('id1', keep=False)
    dups = dups[dups == True].index.values
    for i in dups:
        if 'C' in con_out.loc[i, 'id2']:
            if con_out.loc[i, 'id2'] not in con_out.loc[:, 'id1']:
                swap = con_out.loc[i, 'id1']
                con_out.loc[i, 'id1'] = con_out.loc[i, 'id2']
                con_out.loc[i, 'id2'] = swap
    con_out = con_out.sort_values('id2', ascending=False)
    if branch == 'yes':
        file = 'Branch_'
        os.chdir(r'Output/Branches')
    else:
        file = 'Grid_'
        os.chdir(r'Output/Grids')

    for row in mg_out.iterrows():
        index = grid_resume.loc[
            grid_resume['Cluster'] == int(row[1][0].split('C')[1])].index

        grid_resume.loc[index, 'Connection Length [km]'] = 0
        grid_resume.loc[index, 'Connection Cost [k€]'] = 0
        grid_resume.loc[index, 'Connection Type'] = 'Microgrid'
        grid_resume.loc[index, 'Substation ID'] = 'Microgrid'

#iterate until alle the possible connections are analyzed, start with extreme clusters
    while con_out.empty==False:
        #make a list of all the items of con_out:
        list_items=con_out['id1'].append(con_out['id2']).values.tolist()
        #group items and count their numbers
        my_dict = {i: list_items.count(i) for i in list_items}
        #check elements in my_dict and select the ones present only once
        for k in my_dict:
            if my_dict[k]==1 and 'S' not in k:
                    print(k)
                    break
        index = grid_resume.loc[
            grid_resume['Cluster'] == int(k.split('C')[1])].index
        if os.path.isfile(file + str(k.split('C')[1]) +
                          ".shp"):
            grid_1 = gpd.read_file(file + str(k.split('C')[1]) +
                                   ".shp")
            c_grid_points = list(zip(grid_1.ID1.astype(int),
                                     grid_1.ID2.astype(int)))
            grid_1 = line_to_points(grid_1, geo_df_clustered)
        else:
            grid_1 = geo_df_clustered[
                geo_df_clustered['Cluster'] == int(
                    k.split('C')[1])]
            grid_1 = grid_1[grid_1['Population'] > 0]
            c_grid_points=[]
        # find the second point of the connection
        if (con_out['id1']==k).any():
            k1=con_out[con_out['id1']==k]['id2'].values[0]
        else:
            k1=con_out[con_out['id2']==k]['id1'].values[0]

        if 'C' in k1:
            if os.path.isfile(file + str(k1.split('C')[1]) +
                              ".shp"):
                grid_2 = gpd.read_file(file + str(k1.split('C')[1]) +
                                       ".shp")
                c_grid_points.append(list(zip(grid_2.ID1.astype(int),
                                              grid_2.ID2.astype(int))))
                grid_2 = line_to_points(grid_2, geo_df_clustered)
            else:
                grid_2 = geo_df_clustered[
                    geo_df_clustered['Cluster'] == int(
                        k1.split('C')[1])]
                grid_2 = grid_2[grid_2['Population'] > 0]

            grid_resume.loc[index, 'Connection Type'] = 'Intra cluster connection'

            grid_resume.loc[index, 'Substation ID'] = k1

        elif 'S' in k1:
            sub_in_df = substations[
                substations['ID'] ==
                int(k1.split('S')[1])].nearest_id.values[0]
            grid_2 = geo_df_clustered[geo_df_clustered['ID'] == sub_in_df]

            grid_resume.loc[index, 'Connection Type'] = substations[
                substations['ID'] == int(k1.split('S')[1])].Type.values[
                0]
            grid_resume.loc[index, 'Substation ID'] = substations[
                substations['ID'] == int(k1.split('S')[1])].ID.values[0]

        dist_2d = pd.DataFrame(distance_2d(grid_1, grid_2, 'X', 'Y'),
                               index=grid_1.ID.values,
                               columns=grid_2.ID.values)

        p1 = geo_df_clustered[geo_df_clustered['ID'] == dist_2d.min().idxmin()]
        p2 = geo_df_clustered[geo_df_clustered['ID'] ==
                              dist_2d.min(axis=1).idxmin()]
        # recompute dijsktra on the selected connection, it would be better to save its value from before
        # connection, connection_cost, connection_length, _ = \
        #     dijkstra.dijkstra_connection(geo_df_clustered, p1, p2,
        #                                  c_grid_points, line_bc, resolution)

        connection, connection_cost, connection_length, _ = \
            dijkstra.dijkstra_connection_roads(geo_df_clustered, p1, p2,
                                               c_grid_points, line_bc,
                                               resolution, gdf_roads,
                                               roads_segments)
        if not connection.empty:
            connection.to_file(
                'Connection_' + k.split('C')[1] + '.shp')
        grid_resume.loc[index, 'Connection Length [km]'] = \
            connection_length / 1000
        grid_resume.loc[index, 'Connection Cost [k€]'] = connection_cost / 1000
        print('Connection for Cluster ' + k + ' created')
        total_connections_opt = \
            gpd.GeoDataFrame(pd.concat([total_connections_opt,
                                        connection], sort=True))

        con_out.drop(index=con_out[(((con_out['id1']==k) & (con_out['id2']==k1))|((con_out['id2']==k) & (con_out['id1']==k1)))].index,inplace=True)

    grid_resume.to_csv('grid_resume_opt.csv', index=False)
    if total_connections_opt.empty == False:
        total_connections_opt.crs = geo_df_clustered.crs
        total_connections_opt.to_file('all_connections_opt.shp')
    os.chdir('../..')
    return grid_resume



