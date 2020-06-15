import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.ops import nearest_points
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist

# sys.path.insert(0, 'Codes')  # necessary for native GISEle Packages
# native GISEle Packages


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


# function for long separating lines
def l():
    print('-' * 100)


# function for short separating lines
def s():
    print("-" * 40)


def nearest(row, df, src_column=None):
    """Find the nearest point and return the value from specified column."""
    # Find the geometry that is closest
    nearest_p = df['geometry'] == nearest_points(row['geometry'],
                                                 df.unary_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df.loc[nearest_p, src_column].values[0]
    return value


def distance_2d(df1, df2, x, y):
    """Find the 2D distance matrix between two datasets of points."""

    d1_coordinates = {'x': df1[x], 'y': df1[y]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']

    d2_coordinates = {'x': df2[x], 'y': df2[y]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = distance_matrix(df1_loc, df2_loc)
    return value


def distance_3d(df1, df2, x, y, z):
    """Find the 3D euclidean distance matrix between two datasets of points."""

    d1_coordinates = {'x': df1[x], 'y': df1[y], 'z': df1[z]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']

    d2_coordinates = {'x': df2[x], 'y': df2[y], 'z': df2[z]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = pd.DataFrame(cdist(df1_loc.values, df2_loc.values, 'euclidean'),
                         index=df1_loc.index, columns=df2_loc.index)
    return value


def weight_matrix(gdf, distance_3D, paycheck):

    # Altitude distance in meters
    weight = gdf['Weight'].values
    N = gdf['X'].size
    weight_columns = np.repeat(weight[:, np.newaxis], N, 1)
    weight_rows = np.repeat(weight[np.newaxis, :], N, 0)
    total_weight = (weight_columns + weight_rows) / 2

    # 3D distance
    value = (distance_3D * total_weight) * paycheck / 1000

    return value


def line_to_points(line, df):

    nodes = list(line['ID1'].values) + list(line['ID2'].values)
    nodes = list(dict.fromkeys(nodes))
    nodes_in_df = gpd.GeoDataFrame(crs=df.crs)
    for i in nodes:
        nodes_in_df = nodes_in_df.append(df[df['ID'] == i], sort=False)
    nodes_in_df.reset_index(drop=True, inplace=True)

    return nodes_in_df

def load():
    print("8.Load creation")
    s()
    peop_per_house = int(
        input("How many people are there on average in an household?: "))
    s()

    os.chdir(r'Output//Clusters')
    clusters = gpd.read_file('total_cluster.shp')
    n_clusters = max(clusters['clusters']) + 1
    os.chdir(r'..//..')

    os.chdir(r'Input//3_load_ref')
    clusters_list_2 = list(set(clusters['clusters']))

    load_ref = pd.read_csv('Load.csv', index_col=0)
    users_ref = load_ref.loc[['Numero di utenti per categoria']]
    house_ref = int(
        load_ref['Case "premium"']['Numero di utenti per categoria']
        + load_ref['Case "standard"']['Numero di utenti per categoria'])

    hours = load_ref.index.values
    hours = np.delete(hours, 0, 0)

    os.chdir(r'..//..//Output//Load')
    total_loads = pd.DataFrame(index=hours)

    loads_list = []
    houses_list = []
    for n in clusters_list_2:
        cluster = clusters[clusters['clusters'] == n]
        cluster_pop = sum(cluster['Population'])

        cluster_houses = int(cluster_pop / peop_per_house)

        load_cluster = load_ref.copy()

        public_services = ["Clinica", "Scuola Primaria", "Scuola Secondaria",
                           "Amministrazione"]

        for service in load_cluster.columns.values:

            users_ref = load_ref.loc[['Numero di utenti per categoria']]

            if service in public_services:
                load_cluster.at['Numero di utenti per categoria', service] = 1

            else:

                new_n_users = round(
                    users_ref[service][
                        'Numero di utenti per categoria'] / house_ref * cluster_houses + 0.5,
                    0)
                load_cluster.at[
                    'Numero di utenti per categoria', service] = new_n_users

        for i in hours:
            for j in load_cluster.columns.values:
                new_load = round(load_ref[j][i] / house_ref * cluster_houses,
                                 3)
                load_cluster.at[i, j] = new_load

                users_ref = load_cluster.loc[
                    ['Numero di utenti per categoria']]  # NON CANCELLARE

        name_load = "load_cluster_n_" + str(n) + '_houses_' + str(
            cluster_houses) + '.csv'
        load_cluster.to_csv(name_load)
        loads_list.append(name_load)
        houses_list.append(cluster_houses)
        total_loads[n] = load_cluster[load_cluster.columns].sum(axis=1)

    total_loads.to_csv("total_loads.csv")
    os.chdir(r'..//..')

    print("All loads successfully computed")
    s()

    print("Total load - columns CLUSTERS - rows HOURS")
    print(total_loads)
    l()
    return loads_list, houses_list, house_ref


def sizing(loads_list, clusters_list_2, houses_list, house_ref):
    print("9.Generation dimensioning")
    print(
        "Now you need to use an external tool in order to optimize the power generation block for each cluster.\n"
        "The energy consumption for each of them is in a CSV file named Output/Load/total_loads.csv")
    print(
        "From our side we'll use the Calliope Energy Model in order to optimize the system. This will need you to\n"
        "fill the model with details about each technology. The CSV files are in GISEle\Input\_5_components.\n"
        "At the moment, standard details are already in.")

    'Asking the user info about the project'
    # project_life = int(input("What's the project lifefime?: [years] ")) + 1
    project_life = 25
    time_span = range(project_life)

    os.chdir(r'Output//Clusters')
    clusters = gpd.read_file('total_cluster.shp')
    n_clusters = max(clusters['clusters']) + 1

    clusters_list = np.arange(n_clusters)
    # clusters_list_2 = clusters_list[:]
    clusters_list_2 = list(set(clusters['clusters']))
    os.chdir(r'..//..')

    total_energy = pd.DataFrame(index=clusters_list_2,
                                columns=['Total Energy Consumption [kWh]'])
    mg_NPC = pd.DataFrame(index=clusters_list_2, columns=['Total Cost [$]'])

    i = 0
    'Creating the final load profiles'
    for clusters in loads_list:
        c = clusters_list_2[i]
        n = houses_list[i]
        os.chdir(r'Input//7_sizing_calliope//timeseries_data')

        load_ref = pd.read_csv('demand_power_ref.csv', index_col=0, header=0)
        load_cluster = load_ref.copy()

        load_cluster['X1'] = load_ref['X1'] / house_ref * n
        energy = -1 * load_cluster.values.sum()
        total_energy.at[
            c, 'Total Energy Consumption [kWh]'] = energy * project_life

        load_final = 'whole_' + clusters + '.csv'

        load_cluster.to_csv(load_final)
        load_cluster.to_csv('demand_power.csv')
        os.chdir(r'..//..//..')

        lat = -16.5941
        long = 36.5955

        '''import geopandas as gpd
        import pandas as pd
        from pyproj import transform, Proj
        from fiona.crs import from_epsg
        from shapely.geometry import Point, MultiPoint

        point = gpd.GeoDataFrame(crs=from_epsg(32737))
        point.loc[0, 'x'] = 227911
        point.loc[0, 'y'] = 8097539
        point['geometry'] = None
        point['geometry'].stype = gpd.geoseries.GeoSeries
        point.loc[0, 'geometry'] = Point(point.loc[0, 'x'], point.loc[0, 'y'])

        point['geometry'] = point['geometry'].to_crs(epsg=4326)

        point.loc[0, 'x_deg'] = point.loc[0, 'geometry'].x
        point.loc[0, 'y_deg'] = point.loc[0, 'geometry'].y

        print(point)'''

        pre_calliope(i, project_life, lat, long)

        cost_km_line = 20000
        total_undiscounted_cost = calliope_x_GISEle(time_span, c, cost_km_line,
                                                    project_life)
        l()
        i += 1
        os.chdir(r'..//..')

        mg_NPC.at[c, 'Total Cost [$]'] = total_undiscounted_cost

    return total_energy, mg_NPC


def final_results(clusters_list_2, total_energy, grid_resume, mg_NPC):
    os.chdir(r'Output//Sizing_calliope')
    grid_resume = pd.read_csv('grid_resume.csv', header=0, index_col=0)

    os.chdir(r'..//Clusters')
    clusters = gpd.read_file('total_cluster.shp')
    n_clusters = max(clusters['clusters']) + 1

    clusters_list = np.arange(n_clusters)
    # clusters_list_2 = clusters_list[:]
    clusters_list_2 = list(set(clusters['clusters']))

    os.chdir(r'..//..')

    final_LCOEs = pd.DataFrame(index=clusters_list_2,
                               columns=['GRID_NPC [$]', 'MG_NPC [$]',
                                        'Total Energy Consumption [kWh]',
                                        'MG_LCOE [$/kWh]', 'GRID_LCOE [$/kWh]',
                                        'Suggested Solution'])

    print(total_energy)
    s()
    print(grid_resume)
    s()
    print(mg_NPC)
    s()

    for clu in clusters_list_2:
        print(clu)
        final_LCOEs.at[clu, 'Total Energy Consumption [kWh]'] = \
            total_energy.loc[clu, 'Total Energy Consumption [kWh]']
        final_LCOEs.at[clu, 'GRID_NPC [$]'] = grid_resume.loc[
                                                  clu, 'Grid_Cost'] + \
                                              grid_resume.loc[
                                                  clu, 'Connection_Cost']

        final_LCOEs.at[clu, 'MG_NPC [$]'] = mg_NPC.loc[clu, 'Total Cost [$]']

        mg_lcoe = final_LCOEs.loc[clu, 'MG_NPC [$]'] / final_LCOEs.loc[
            clu, 'Total Energy Consumption [kWh]']

        final_LCOEs.at[clu, 'MG_LCOE [$/kWh]'] = mg_lcoe

        grid_lcoe = final_LCOEs.loc[clu, 'GRID_NPC [$]'] / final_LCOEs.loc[
            clu, 'Total Energy Consumption [kWh]'] + 0.1428

        final_LCOEs.at[clu, 'GRID_LCOE [$/kWh]'] = grid_lcoe

        if mg_lcoe > grid_lcoe:
            ratio = grid_lcoe / mg_lcoe
            if ratio < 0.9:
                proposal = 'GRID'
            else:
                proposal = 'Although the GRID option is cheaper, ' \
                           'the two costs are very similar, so further investigation is suggested'

        else:
            ratio = mg_lcoe / grid_lcoe
            if ratio < 0.9:
                proposal = 'OFF-GRID'
            else:
                proposal = 'Although the OFF-GRID option is cheaper, ' \
                           'the two costs are very similar, so further investigation is suggested'

        final_LCOEs.at[clu, 'Suggested Solution'] = proposal

    l()
    print(final_LCOEs)
    l()

    os.chdir(r'Output//Final_results')
    final_LCOEs.to_csv('final_LCOE.csv')

    return final_LCOEs
