import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import shapely.ops
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from shapely.geometry import Point, box, LineString


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
    nearest_p = df['geometry'] == shapely.ops.nearest_points(row['geometry'],
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


def weight_matrix(gdf, dist_3d_matrix, line_bc):
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
    nodes = list(line.ID1.astype(int)) + list(line.ID2.astype(int))
    nodes = list(dict.fromkeys(nodes))
    nodes_in_df = gpd.GeoDataFrame(crs=df.crs)
    for i in nodes:
        nodes_in_df = nodes_in_df.append(df[df['ID'] == i], sort=False)
    nodes_in_df.reset_index(drop=True, inplace=True)

    return nodes_in_df


def create_box(limits, df):
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
        extension = dist / 8

    bubble = box(minx=x_min - extension, maxx=x_max + extension,
                 miny=y_min - extension, maxy=y_max + extension)
    df_box = df[df.within(bubble)]
    df_box.index = pd.Series(range(0, len(df_box['ID'])))

    return df_box


def edges_to_line(path, df, edges_matrix):
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


def load(clusters_list):
    l()
    print("5. Microgrid Sizing")
    l()

    os.chdir(r'Input//')
    print("Creating load profile for each cluster..")
    load_profile = pd.DataFrame(index=range(0, 24),
                                columns=clusters_list.Cluster)
    input_profile = pd.read_csv('Load Profile.csv')
    for column in load_profile:
        load_profile.loc[:, column] = \
            (input_profile.loc[:, 'Hourly Factor']
             * clusters_list.loc[column, 'Load']).values

    print("Load profile created")
    return load_profile


def sizing(load_profile, clusters_list):

    mg_energy = pd.DataFrame(load_profile)
    mg_npc = pd.DataFrame(clusters_list)

    return mg_energy, mg_npc


def final_results(clusters_list, total_energy, grid_resume, mg_npc):

    os.chdir(r'..//..')

    final_lcoe = pd.DataFrame(index=clusters_list,
                              columns=['GRID_NPC [$]', 'MG_NPC [$]',
                                       'Total Energy Consumption [kWh]',
                                       'MG_LCOE [$/kWh]', 'GRID_LCOE [$/kWh]',
                                       'Suggested Solution'])

    print(total_energy)
    s()
    print(grid_resume)
    s()
    print(mg_npc)
    s()

    for i in clusters_list:
        print(i)
        final_lcoe.at[i, 'Total Energy Consumption [kWh]'] = \
            total_energy.loc[i, 'Total Energy Consumption [kWh]']
        final_lcoe.at[i, 'GRID_NPC [$]'] = \
            grid_resume.loc[i, 'Grid_Cost'] \
            + grid_resume.loc[i, 'Connection_Cost']

        final_lcoe.at[i, 'MG_NPC [$]'] = mg_npc.loc[i, 'Total Cost [$]']

        mg_lcoe = final_lcoe.loc[i, 'MG_NPC [$]'] / final_lcoe.loc[
            i, 'Total Energy Consumption [kWh]']

        final_lcoe.at[i, 'MG_LCOE [$/kWh]'] = mg_lcoe

        grid_lcoe = final_lcoe.loc[i, 'GRID_NPC [$]'] / final_lcoe.loc[
            i, 'Total Energy Consumption [kWh]'] + 0.1428

        final_lcoe.at[i, 'GRID_LCOE [$/kWh]'] = grid_lcoe

        if mg_lcoe > grid_lcoe:
            ratio = grid_lcoe / mg_lcoe
            if ratio < 0.9:
                proposal = 'GRID'
            else:
                proposal = 'Although the GRID option is cheaper, ' \
                           'the two costs are very similar, so further ' \
                           'investigation is suggested'

        else:
            ratio = mg_lcoe / grid_lcoe
            if ratio < 0.9:
                proposal = 'OFF-GRID'
            else:
                proposal = 'Although the OFF-GRID option is cheaper, ' \
                           'the two costs are very similar, so further ' \
                           'investigation is suggested'

        final_lcoe.at[i, 'Suggested Solution'] = proposal

    l()
    print(final_lcoe)
    l()

    os.chdir(r'Output//Final_results')
    final_lcoe.to_csv('final_LCOE.csv')

    return final_lcoe
