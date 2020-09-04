"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Supporting Code

Group of supporting functions used inside all the process of GISEle algorithm
"""

import os
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely.ops
import iso8601
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from shapely.geometry import Point, box, LineString
from Codes.michele.Michele_run import start
from Codes.data_import import import_pv_data, import_wind_data


def l():
    """Print long separating lines."""
    print('-' * 100)


def s():
    """Print short separating lines."""
    print("-" * 40)


def nearest(row, df, src_column=None):
    """
    Find the nearest point and return the value from specified column.
    :param row: Iterative row of the first dataframe
    :param df: Second dataframe to be found the nearest value
    :param src_column: Column of the second dataframe that will be returned
    :return value: Value of the desired src_column of the second dataframe
    """

    # Find the geometry that is closest
    nearest_p = df['geometry'] == shapely.ops.nearest_points(row['geometry'],
                                                             df.unary_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df.loc[nearest_p, src_column].values[0]
    return value


def distance_2d(df1, df2, x, y):
    """
    Find the 2D distance matrix between two datasets of points.
    :param df1: first point dataframe
    :param df2: second point dataframe
    :param x: column representing the x coordinates (longitude)
    :param y: column representing the y coordinates (latitude)
    :return value: 2D Distance matrix between df1 and df2
    """

    d1_coordinates = {'x': df1[x], 'y': df1[y]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']

    d2_coordinates = {'x': df2[x], 'y': df2[y]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = distance_matrix(df1_loc, df2_loc)
    return value


def distance_3d(df1, df2, x, y, z):
    """
    Find the 3D euclidean distance matrix between two datasets of points.
    :param df1: first point dataframe
    :param df2: second point dataframe
    :param x: column representing the x coordinates (longitude)
    :param y: column representing the y coordinates (latitude)
    :param z: column representing the z coordinates (Elevation)
    :return value: 3D Distance matrix between df1 and df2
    """

    d1_coordinates = {'x': df1[x], 'y': df1[y], 'z': df1[z]}
    df1_loc = pd.DataFrame(data=d1_coordinates)
    df1_loc.index = df1['ID']

    d2_coordinates = {'x': df2[x], 'y': df2[y], 'z': df2[z]}
    df2_loc = pd.DataFrame(data=d2_coordinates)
    df2_loc.index = df2['ID']

    value = pd.DataFrame(cdist(df1_loc.values, df2_loc.values, 'euclidean'),
                         index=df1_loc.index, columns=df2_loc.index)
    return value


def cost_matrix(gdf, dist_3d_matrix, line_bc):
    """
    Creates the cost matrix in €/km by finding the average weight between
    two points and then multiplying by the distance and the line base cost.
    :param gdf: Geodataframe being analyzed
    :param dist_3d_matrix: 3D distance matrix of all points [meters]
    :param line_bc: line base cost for line deployment [€/km]
    :return value: Cost matrix of all the points present in the gdf [€]
    """
    # Altitude distance in meters
    weight = gdf['Weight'].values
    n = gdf['X'].size
    weight_columns = np.repeat(weight[:, np.newaxis], n, 1)
    weight_rows = np.repeat(weight[np.newaxis, :], n, 0)
    total_weight = (weight_columns + weight_rows) / 2

    # 3D distance
    value = (dist_3d_matrix * total_weight) * line_bc / 1000

    return value


def line_to_points(line, df):
    """
    Finds all the points of a linestring geodataframe correspondent to a
    point geodataframe.
    :param line: Linestring geodataframe being analyzed
    :param df: Point geodataframe where the linestring nodes will be referenced
    :return nodes_in_df: Point geodataframe containing all nodes of linestring
    """
    nodes = list(line.ID1.astype(int)) + list(line.ID2.astype(int))
    nodes = list(dict.fromkeys(nodes))
    nodes_in_df = gpd.GeoDataFrame(crs=df.crs, geometry=[])
    for i in nodes:
        nodes_in_df = nodes_in_df.append(df[df['ID'] == i], sort=False)
    nodes_in_df.reset_index(drop=True, inplace=True)

    return nodes_in_df


def create_box(limits, df):
    """
    Creates a delimiting box around a geodataframe.
    :param limits: Linestring geodataframe being analyzed
    :param df: Point geodataframe to be delimited
    :return df_box: All points of df that are inside the delimited box
    """
    x_min = min(limits.X)
    x_max = max(limits.X)
    y_min = min(limits.Y)
    y_max = max(limits.Y)

    dist = Point(x_min, y_min).distance(Point(x_max, y_max))
    if dist < 5000:
        extension = dist * 2
    elif dist < 15000:
        extension = dist
    else:
        extension = dist / 4

    bubble = box(minx=x_min - extension, maxx=x_max + extension,
                 miny=y_min - extension, maxy=y_max + extension)
    df_box = df[df.within(bubble)]
    df_box.index = pd.Series(range(0, len(df_box['ID'])))

    return df_box


def edges_to_line(path, df, edges_matrix):
    """
    Transforms a list of NetworkX graph edges into a linestring geodataframe
    based on a input point geodataframe
    :param path: NetworkX graph edges sequence
    :param df: Point geodataframe to be used as reference
    :param edges_matrix: Matrix containing the cost to connect a pair of points
    :return line: Linestring geodataframe containing point IDs and its cost
    :return line_points: All points of df that are part of the linestring
    """
    steps = len(path)
    line = gpd.GeoDataFrame(index=range(0, steps),
                            columns=['ID1', 'ID2', 'Cost', 'geometry'],
                            crs=df.crs)
    line_points = []
    for h in range(0, steps):
        line.at[h, 'geometry'] = LineString(
            [(df.loc[path[h][0], 'X'],
              df.loc[path[h][0], 'Y']),
             (df.loc[path[h][1], 'X'],
              df.loc[path[h][1], 'Y'])])

        # int here is necessary to use the command .to_file
        line.at[h, 'ID1'] = int(df.loc[path[h][0], 'ID'])
        line.at[h, 'ID2'] = int(df.loc[path[h][1], 'ID'])
        line.at[h, 'Cost'] = int(edges_matrix.loc[df.loc[path[h][0], 'ID'],
                                                  df.loc[path[h][1], 'ID']])
        line_points.append(list(df.loc[path[h], 'ID']))
    line.drop(line[line['Cost'] == 0].index, inplace=True)
    line.Cost = line.Cost.astype(int)
    return line, line_points


def load(clusters_list, grid_lifetime):
    """
    Reads the input daily load profile from the input csv. Reads the number of
    years of the project and the demand growth from the data.dat file of
    Micehele. Then it multiplies the load profile by the Clusters' peak load
    and append values to create yearly profile composed of 12 representative
    days.
    :param grid_lifetime: Number of years the grid will operate
    :param clusters_list: List of clusters ID numbers
    :return load_profile: Cluster load profile for the whole period
    :return years: Number of years the microgrid will operate
    :return total_energy: Energy provided by the grid in its lifetime [kWh]
    """
    l()
    print("5. Microgrid Sizing")
    l()
    data_michele = pd.read_table("Codes/michele/Inputs/data.dat", sep="=",
                                 header=None)
    os.chdir(r'Input//')
    print("Creating load profile for each cluster..")
    daily_profile = pd.DataFrame(index=range(1, 25),
                                 columns=clusters_list.Cluster)
    input_profile = pd.read_csv('Load Profile.csv')
    for column in daily_profile:
        daily_profile.loc[:, column] = \
            (input_profile.loc[:, 'Hourly Factor']
             * clusters_list.loc[column, 'Load']).values
    rep_days = int(data_michele.loc[0, 1].split(';')[0])
    grid_energy = daily_profile.append([daily_profile] * 364,
                                       ignore_index=True)
    #  append 11 times since we are using 12 representative days in a year
    load_profile = daily_profile.append([daily_profile] * (rep_days - 1),
                                        ignore_index=True)

    years = int(data_michele.loc[1, 1].split(';')[0])
    demand_growth = float(data_michele.loc[87, 1].split(';')[0])
    daily_profile_new = daily_profile
    #  appending for all the years considering demand growth
    for i in range(grid_lifetime - 1):
        daily_profile_new = daily_profile_new.multiply(1 + demand_growth)
        if i < (years - 1):
            load_profile = load_profile.append([daily_profile_new] * rep_days,
                                               ignore_index=True)
        grid_energy = grid_energy.append([daily_profile_new] * 365,
                                         ignore_index=True)
    total_energy = pd.DataFrame(index=clusters_list.Cluster,
                                columns=['Energy'])
    for cluster in clusters_list.Cluster:
        total_energy.loc[cluster, 'Energy'] = \
            grid_energy.loc[:, cluster].sum().round(2)
    os.chdir('..')
    print("Load profile created")
    total_energy.to_csv(r'Output/Microgrids/Grid_energy.csv')
    return load_profile, years, total_energy


def shift_timezone(df, shift):
    """
    Move the values of a dataframe with DateTimeIndex to another UTC zone,
    adding or removing hours.
    :param df: Dataframe to be analyzed
    :param shift: Amount of hours to be shifted
    :return df: Input dataframe with values shifted in time
    """
    if shift > 0:
        add_hours = df.tail(shift)
        df = pd.concat([add_hours, df], ignore_index=True)
        df.drop(df.tail(shift).index, inplace=True)
    elif shift < 0:
        remove_hours = df.head(abs(shift))
        df = pd.concat([df, remove_hours], ignore_index=True)
        df.drop(df.head(abs(shift)).index, inplace=True)
    return df


def sizing(load_profile, clusters_list, geo_df_clustered, wt, years):
    """
    Imports the solar and wind production from the RenewablesNinja api and then
    Runs the optimization algorithm MicHEle to find the best microgrid
    configuration for each Cluster.
    :param load_profile: Load profile of all clusters during all years
    :param clusters_list: List of clusters ID numbers
    :param geo_df_clustered: Point geodataframe with Cluster identification
    :param wt: Wind turbine model used for computing the wind velocity
    :param years: Number of years the microgrid will operate (project lifetime)
    :return mg: Dataframe containing the information of the Clusters' microgrid
    """
    geo_df_clustered = geo_df_clustered.to_crs(4326)
    mg = pd.DataFrame(index=clusters_list.index,
                      columns=['PV', 'Wind', 'Diesel', 'BESS', 'Inverter',
                               'Investment Cost', 'OM Cost', 'Replace Cost',
                               'Total Cost', 'Energy Produced',
                               'Energy Consumed', 'LCOE'])
    for cluster_n in clusters_list.Cluster:

        l()
        print('Creating the optimal Microgrid for Cluster ' + str(cluster_n))
        l()
        load_profile_cluster = load_profile.loc[:, cluster_n]
        lat = geo_df_clustered[geo_df_clustered['Cluster']
                               == cluster_n].geometry.y.values[0]
        lon = geo_df_clustered[geo_df_clustered['Cluster']
                               == cluster_n].geometry.x.values[0]
        all_angles = pd.read_csv('Input/TiltAngles.csv')
        tilt_angle = abs(all_angles.loc[abs(all_angles['lat'] - lat).idxmin(),
                                        'opt_tilt'])
        pv_prod = import_pv_data(lat, lon, tilt_angle)
        utc = pv_prod.local_time[0]
        if type(utc) is pd.Timestamp:
            time_shift = utc.hour
        else:
            utc = iso8601.parse_date(utc)
            time_shift = int(utc.tzinfo.tzname(utc).split(':')[0])
        pv_avg = pv_prod.groupby([pv_prod.index.month,
                                  pv_prod.index.hour]).mean()
        pv_avg = pv_avg.append([pv_avg] * (years - 1), ignore_index=True)
        pv_avg.reset_index(drop=True, inplace=True)
        pv_avg = shift_timezone(pv_avg, time_shift)

        wt_prod = import_wind_data(lat, lon, wt)
        wt_avg = wt_prod.groupby([wt_prod.index.month,
                                  wt_prod.index.hour]).mean()
        wt_avg = wt_avg.append([wt_avg] * (years - 1), ignore_index=True)
        wt_avg.reset_index(drop=True, inplace=True)
        wt_avg = shift_timezone(wt_avg, time_shift)

        inst_pv, inst_wind, inst_dg, inst_bess, inst_inv, init_cost, \
            rep_cost, om_cost, salvage_value, gen_energy, load_energy = \
            start(load_profile_cluster, pv_avg, wt_avg)

        mg.loc[cluster_n, 'PV'] = inst_pv
        mg.loc[cluster_n, 'Wind'] = inst_wind
        mg.loc[cluster_n, 'Diesel'] = inst_dg
        mg.loc[cluster_n, 'BESS'] = inst_bess
        mg.loc[cluster_n, 'Inverter'] = inst_inv
        mg.loc[cluster_n, 'Investment Cost'] = init_cost
        mg.loc[cluster_n, 'OM Cost'] = om_cost
        mg.loc[cluster_n, 'Replace Cost'] = rep_cost
        mg.loc[cluster_n, 'Total Cost'] = rep_cost + om_cost + init_cost
        mg.loc[cluster_n, 'Energy Produced'] = gen_energy
        mg.loc[cluster_n, 'Energy Consumed'] = load_energy
        mg.loc[cluster_n, 'LCOE'] = (rep_cost + om_cost + init_cost)/gen_energy
        print(mg)

    mg.to_csv('Output/Microgrids/microgrids.csv', index_label='Cluster')

    return mg


def download_url(url, out_path, chunk_size=128):
    """
    Download zip file from specified url and save it into a defined folder
    Note: check chunk_size parameter
    """
    r = requests.get(url, stream=True)
    with open(out_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def download_tif(area, crs, scale, image, out_path):
    """
    Download data from Earth Engine
    :param area: GeoDataFrame with the polygon of interest area
    :param crs: str with crs of the project
    :param scale: int with pixel size in meters
    :param image: image from the wanted database in Earth Image
    :param out_path: str with output path
    :return:
    """
    min_x, min_y, max_x, max_y = area.geometry.total_bounds
    path = image.getDownloadUrl({
        'scale': scale,
        'crs': 'EPSG:' + str(crs),
        'region': [[min_x, min_y], [min_x, max_y], [max_x, min_y],
                   [max_x, max_y]]
    })
    print(path)
    download_url(path, out_path)
    return


def lcoe_analysis(clusters_list, total_energy, grid_resume, mg, coe,
                  grid_ir, grid_om, grid_lifetime):
    """
     Computes the LCOE of the on-grid and off-grid and compares both of them
     to find the best solution.
     :param clusters_list: List of clusters ID numbers
     :param total_energy: Energy provided by the grid in its lifetime [kWh]
     :param grid_resume: Table summarizing grid and connection costs
     :param mg: Table summarizing all the microgrids and its costs.
     :param coe: Cost of electricity in the market [€/kWh]
     :param grid_ir: Inflation rate for the grid investments [%]
     :param grid_om: Operation and maintenance cost of grid [% of invest cost]
     :param grid_lifetime: Number of years the grid will operate
     :return final_lcoe: Table summary of all LCOEs and the proposed solution
     """

    final_lcoe = pd.DataFrame(index=clusters_list.Cluster,
                              columns=['Grid NPC [EUR]', 'MG NPC [EUR]',
                                       'Grid Energy Consumption [kWh]',
                                       'MG LCOE [EUR/kWh]',
                                       'Grid LCOE [EUR/kWh]', 'Best Solution'])
    for i in clusters_list.Cluster:
        final_lcoe.at[i, 'Grid Energy Consumption [kWh]'] = \
            total_energy.loc[i, 'Energy']

        # finding the npv of the cost of O&M for the whole grid lifetime
        total_grid_om = grid_om * grid_resume.loc[i, 'Grid Cost'] * 1000
        total_grid_om = [total_grid_om]*grid_lifetime
        total_grid_om = np.npv(grid_ir, total_grid_om)

        final_lcoe.at[i, 'Grid NPC [EUR]'] = \
            (grid_resume.loc[i, 'Grid Cost'] +
             grid_resume.loc[i, 'Connection Cost']) * 1000 \
            + total_grid_om

        final_lcoe.at[i, 'MG NPC [EUR]'] = mg.loc[i, 'Total Cost'] * 1000

        final_lcoe.at[i, 'MG LCOE [EUR/kWh]'] = mg.loc[i, 'LCOE']

        grid_lcoe = final_lcoe.loc[i, 'Grid NPC [EUR]'] / final_lcoe.loc[
            i, 'Grid Energy Consumption [kWh]'] + coe

        final_lcoe.at[i, 'Grid LCOE [EUR/kWh]'] = grid_lcoe

        if mg.loc[i, 'LCOE'] > grid_lcoe:
            ratio = grid_lcoe / mg.loc[i, 'LCOE']
            if ratio < 0.95:
                proposal = 'GRID'
            else:
                proposal = 'BOTH'

        else:
            ratio = mg.loc[i, 'LCOE'] / grid_lcoe
            if ratio < 0.95:
                proposal = 'OFF-GRID'
            else:
                proposal = 'BOTH'

        final_lcoe.at[i, 'Best Solution'] = proposal

    l()
    print(final_lcoe)

    final_lcoe.to_csv(r'Output/Microgrids/LCOE_Analysis.csv')

    return final_lcoe
