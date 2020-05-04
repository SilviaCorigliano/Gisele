import os
import glob
import pandas as pd
import math
import geopandas as gpd
import networkx as nx
import numpy as np
from fiona.crs import from_epsg
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from shapely.geometry import Point, MultiPoint, box, LineString
from shapely.ops import nearest_points
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from shapely.ops import nearest_points
from Codes.Weight_matrix import *
from Codes.remove_cycles_weight_final import *
import warnings
#  remove the warnings from the output screen
warnings.filterwarnings("ignore")
import sys

sys.path.insert(0, 'Codes')  # necessary for native GISEle Packages
# native GISEle Packages
from Codes.Steinerman import *
from Codes.Spiderman import *

def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


# function for long separating lines
def l():
    print('-' * 100)


# function for short separating lines
def s():
    print("-" * 40)


def import_csv_file():
    print("3. Importing and processing base layer - Let's import the necessary layer!")
    os.chdir(r'Input//1_starting_CSV_files')
    print("The available csv files in this directory are: ")
    csv_files = glob.glob('*.{}'.format('csv'))
    s()
    print("\n".join(csv_files))
    s()
    file_name = str(input("Which file do you want to load?: "))
    points = pd.read_csv(file_name + '.csv', sep=',')
    print("Layer file successfully loaded.")
    s()

    proj_coords = int(input("Provide crs code of your layer reference system: "))
    s()

    unit = input("Enter coordinate measurement unit used by your crs: 0 = degrees, 1 = meters? ")
    s()
    l()
    resolution = int(input("Enter the resolution of the grid in the measurement unit chosen: "))
    os.chdir(r'..//..')
    return points, proj_coords, unit, file_name, resolution


def import_weighted_file():
    print("5. Importing and processing base layer - Let's import the necessary layer!")
    os.chdir(r'Input//2_weighted_datasets')
    print("The available shp files in this directory are: ")
    shp_files = glob.glob('*.{}'.format('csv'))
    s()
    print("\n".join(shp_files))
    s()
    # file_name = str(input("Which file do you want to load?: "))
    file_name = 'enel_studyarea3_weighted'
    geo_csv = pd.read_csv(file_name + ".csv")

    print("Layer file successfully loaded.")
    s()
    geometry = [Point(xy) for xy in zip(geo_csv.X, geo_csv.Y)]
    # proj_coords = int(input("Provide crs code of your layer reference system: "))
    proj_coords = 32723
    geo_DataFrame = gpd.GeoDataFrame(geo_csv, crs=from_epsg(proj_coords), geometry=geometry)
    s()
    # unit = input("Enter coordinate measurement unit used by your crs: 0 = degrees, 1 = meters? ")
    unit = 1
    s()
    l()
    # resolution = int(input("Enter the resolution of the grid in the measurement unit chosen: "))
    resolution = 1000
    os.chdir(r'..//..')
    return geo_DataFrame, proj_coords, unit, file_name, resolution


def Clean_and_Weight(points):
    N = points.__len__()
    # Cleaning section
    points_clean = points.dropna(subset=['Elevation'])
    points_clean.reset_index(drop=True)
    points_clean.Slope.fillna(value=0, inplace=True)
    points_clean.rename(columns={'Landcover': 'Land_cover'}, inplace=True)
    points_clean.rename(columns={'Riverflow': 'River_flow'}, inplace=True)
    points_clean.rename(columns={'NAME': 'Water_body'}, inplace=True)
    points_clean.Land_cover.fillna(method='bfill', inplace=True)
    points_clean['Land_cover'] = points_clean['Land_cover'].round(0)
    points_clean.Population.fillna(value=0, inplace=True)
    points_clean['Weight'] = 0
    print('Pulizia è stata fatta!')
    print('Sto creando i pesi punto per punto...')
    # Weighting section
    for index, row in points_clean.iterrows():
        # print('iteration', index + 1, 'of', N)
        # Slope conditions
        points_clean.loc[index, 'Weight'] = row.Weight + math.exp(0.01732867951 * row.Slope)
        # Land cover
        if row.Land_cover == 1 or row.Land_cover == 3 or row.Land_cover == 11 or row.Land_cover == 12 \
                or row.Land_cover == 13 or row.Land_cover == 14 or row.Land_cover == 16 or row.Land_cover == 17 \
                or row.Land_cover == 18 or row.Land_cover == 19 or row.Land_cover == 22:
            points_clean.loc[index, 'Weight'] += 1
        elif row.Land_cover == 2:
            points_clean.loc[index, 'Weight'] += 5
        elif row.Land_cover == 7 or row.Land_cover == 8:
            points_clean.loc[index, 'Weight'] += 8
        elif row.Land_cover == 4 or row.Land_cover == 5 or row.Land_cover == 6:
            points_clean.loc[index, 'Weight'] += 3
        elif row.Land_cover == 9 or row.Land_cover == 10:
            points_clean.loc[index, 'Weight'] += 2
        elif row.Land_cover == 15:
            points_clean.loc[index, 'Weight'] += 6
        elif row.Land_cover == 20:
            points_clean.loc[index, 'Weight'] += 9
        elif row.Land_cover == 21:
            points_clean.loc[index, 'Weight'] += 7
        # Water Bodies
        if pd.isna(row.Water_body):
            points_clean.loc[index, 'Weight'] += 0
        else:
            points_clean.loc[index, 'Weight'] += 10
        # Restricted Areas
        # if pd.isna(row.Protected_areas):
        #    points_clean.loc[index, 'Weight'] += 0
        # else:
        #    points_clean.loc[index, 'Weight'] += 99999999
        # Rivers
        if abs(row.River_flow) > 10:
            points_clean.loc[index, 'Weight'] += 9
        else:
            points_clean.loc[index, 'Weight'] += 0
        # Road distance conditions
        # Road distance rules!!! Ground characteristics don't matter if road is really close to the site
        if row.Road_dist < 100:
            points_clean.loc[index, 'Weight'] = 1
        elif row.Road_dist < 1000:
            points_clean.loc[index, 'Weight'] += 5 * row.Road_dist / 1000
        else:
            points_clean.loc[index, 'Weight'] += 5

    points_clean.drop_duplicates(['ID'], keep='last', inplace=True)
    points_clean = points_clean.drop(['Slope', 'Land_cover', 'River_flow', 'Water_body', 'Road_dist'], axis=1)
    print('Weighting process terminated')
    s()
    return points_clean


def carwash(points):
    print("4. Assigning weights, Cleaning and creating the DataFrame")

    points_clean = Clean_and_Weight(points)
    df_in = points_clean
    del points_clean
    # remove not useful columns
    valid_fields = ['ID', 'X', 'Y', 'Population', 'Elevation', 'Weight']
    blacklist = []
    for x in df_in.columns:
        if x not in valid_fields:
            blacklist.append(x)
    df_in.drop(blacklist, axis=1, inplace=True)
    # remove repetitions in the dataframe
    df_in.drop_duplicates(['ID'], keep='last', inplace=True)

    print("Cleaning and weighting process completed")
    s()
    return df_in


def creating_dataframe(Point, df_in, proj_coords, unit, file_name):
    print("Creation of a GeoDataFrame")
    geometry = [Point(xy) for xy in zip(df_in['X'], df_in['Y'])]
    geodf_in = gpd.GeoDataFrame(df_in, geometry=geometry, crs=from_epsg(proj_coords))
    # - Check crs conformity for the clustering procedure
    if unit == '0':
        print('Attention: the clustering procedure requires measurements in meters. '
              'Therefore, your dataset must be reprojected.')
        Proj_coords = int(input("Provide a valid crs code to which reproject:"))
        geodf_in['geometry'] = geodf_in['geometry'].to_crs(epsg=Proj_coords)
        print("Done")
    del df_in
    print("DataFrame correctly created.")
    s()
    choice_save = str(input("Do you want to save the processed file as a .CSV and .SHP? (y/n): "))
    if choice_save == "y":
        os.chdir(r'Input//2_weighted_datasets')
        geodf_in.to_csv(file_name + "_weighted.csv")
        geodf_in.to_file(file_name + "_weighted.shp")
        os.chdir(r'..//..')
        print("Both files successfully saved.")
    l()
    return geodf_in


def info_dataframe(geodf_in):
    print("5.DataFrame preparation")

    total_points = len(geodf_in)
    total_people = int(geodf_in['Population'].sum(axis=0))

    print("In the considered area there are " + str(total_points) + " points, for a total of " + str(total_people) +
          " people which could gain access to electricity.")
    return total_points, total_people


def dataframe_3D(geodf_in):
    d = {'x': geodf_in['X'], 'y': geodf_in['Y'], 'z': geodf_in['Elevation']}
    pop_points = pd.DataFrame(data=d)
    pop_points.index = geodf_in['ID']
    l()
    return pop_points


def clustering_sensitivity(pop_points, geodf_in, total_points, total_people):
    print("6.Clustering - Searching for more densely populated areas")
    print("To do this we need two important parameters defining what's a dense area.\n"
          "This passes through the definition of the CORE area which is a zone sufficiently big and sufficiently "
          "populated to be considered as valuable for electrification.\n"
          "We'll need you to define 'RANGE' and 'SPANS' for the two parameters defining these CORE areas,"
          " which are the MAX RADIUS and the MINIMUM POPULATION.")

    spans = "n"
    while spans == "n":

        s()
        eps_min = float(input("Which is the lower limit for the MAX RADIUS (in meters) you want to consider?: "))
        s()
        eps_max = float(input("Which is the upper limit for the MAX RADIUS (in meters) you want to consider?: "))
        s()
        # d1 = int(input("How many SPANS between these values?: "))
        d1 = 10
        s()
        min_people_min = int(input("Which is the lower limit for the MINIMUM POPULATION you want to consider?: "))
        s()
        min_people_max = int(input("Which is the upper limit for the MINIMUM POPULATION you want to consider?: "))
        s()
        # d2 = int(input("How many SPANS between these values?: "))
        d2 = 10
        s()
        # eps_min = 800
        # eps_max = 2000
        # min_people_min = 100
        # min_people_max = 700
        feature_matrix = pop_points.values
        pop_weight = geodf_in['Population']

        span_eps = []
        for j in range(1, d1 + 1):  # eps
            if d1 != 1:
                eps1 = eps_min + (eps_max - eps_min) / (d1 - 1) * (j - 1)
            else:
                eps1 = eps_min + (eps_max - eps_min) / d1 * (j - 1)
            span_eps.append(eps1)

        span_pop = []
        for i in range(1, d2 + 1):  # min_people
            if d2 != 1:
                peop1 = min_people_min + (min_people_max - min_people_min) / (d2 - 1) * (i - 1)
            else:
                peop1 = min_people_min + (min_people_max - min_people_min) / d2 * (i - 1)
            span_pop.append(peop1)

        tab_cluster = pd.DataFrame(index=span_eps, columns=span_pop)
        tab_peop = pd.DataFrame(index=span_eps, columns=span_pop)
        tab_area = pd.DataFrame(index=span_eps, columns=span_pop)
        tab_people_area = pd.DataFrame(index=span_eps, columns=span_pop)
        for eps1 in span_eps:  # eps

            for peop1 in span_pop:  # min_people

                'Clustering algorithm'
                db = DBSCAN(eps=eps1, min_samples=peop1, metric='euclidean').fit(feature_matrix,
                                                                                 sample_weight=pop_weight)

                core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
                core_samples_mask[db.core_sample_indices_] = True
                labels = db.labels_

                n_clusters_ = int(len(set(labels)) - (1 if -1 in labels else 0))
                #                n_noise_ = list(labels).count(-1)
                gdf_clusters = geodf_in
                gdf_clusters['clusters'] = labels
                gdf_clusters = gdf_clusters.sort_values(by=['ID'])
                gdf_pop_noise = gdf_clusters[gdf_clusters['clusters'] == -1]

                noise_people = round(sum(gdf_pop_noise['Population']), 0)
                area = total_points - len(gdf_pop_noise)
                perc_area = int(area / total_points * 100)
                clustered_people = int((1 - noise_people / total_people) * 100)
                people_area = (total_people - noise_people)/area

                tab_cluster.at[eps1, peop1] = n_clusters_
                tab_peop.at[eps1, peop1] = clustered_people
                tab_area.at[eps1, peop1] = perc_area
                tab_people_area.at[eps1, peop1] = people_area

        print("Number of clusters - columns MINIMUM POPULATION - rows MAX RADIUS")
        print(tab_cluster)
        s()
        print("% of clustered people - columns MINIMUM POPULATION - rows MAX RADIUS")
        print(tab_peop)
        s()
        print("% of clustered area - columns MINIMUM POPULATION - rows MAX RADIUS")
        print(tab_area)
        s()
        print("People per area - columns MINIMUM POPULATION - rows MAX RADIUS")
        print(tab_people_area)
        s()
        spans = str(input("Are you satisfied or want to try again circumscribing a more precise interval? (y/n): "))
        s()

    print("Clustering sensitivity process completed!")
    s()
    export = str(input("Do you want to export CSV files of these tables? (y/n): "))
    s()
    if export == "y":
        os.chdir(r'Output//Sensitivity')
        tab_cluster.to_csv("n_clusters.csv")
        tab_peop.to_csv("%_peop.csv")
        tab_area.to_csv("%_area.csv")
        tab_people_area.to_csv("people_area.csv")
        os.chdir(r'..//..')
    l()

    return feature_matrix, pop_weight


def import_cluster():
    os.chdir(r'Output//Clusters')
    gdf_clusters = gpd.read_file("gdf_clusters.shp")
    clusters_list_2 = np.unique(gdf_clusters['clusters'])  # takes all unique values inside the dataframe
    clusters_list_2 = np.delete(clusters_list_2, np.where(clusters_list_2 == -1))  # removes the noises
    clusters_list = np.vstack([clusters_list_2, clusters_list_2, clusters_list_2])
    # pop_load = int(input("What is the load estimation in kW per person?"))
    pop_load = 0.4  # estimation of 2kW per person
    for i in clusters_list_2:
        pop_cluster = gdf_clusters[gdf_clusters['clusters'] == i]
        clusters_list[1, np.where(clusters_list_2 == i)] = sum(pop_cluster['Population'])
        clusters_list[2, np.where(clusters_list_2 == i)] = clusters_list[1, np.where(clusters_list_2 == i)] * pop_load

    print("Clusters successfully imported")
    l()
    os.chdir(r'..//..')
    return gdf_clusters, clusters_list_2, clusters_list


def clustering(feature_matrix, pop_weight, geodf_in, total_people):
    print("Choose the final combination of MINIMUM POPULATION and MAX RADIUS")
    s()
    eps = float(input("MAX RADIUS: "))
    s()
    min_people = int(input("MINIMUM POPULATION: "))
    s()

    db = DBSCAN(eps=eps, min_samples=min_people, metric='euclidean').fit(feature_matrix, sample_weight=pop_weight)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    labels_2 = labels[:]

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    # create output shapefile with additional columns for clusters
    gdf_clusters = geodf_in
    gdf_clusters['clusters'] = labels
    gdf_clusters = gdf_clusters.sort_values(by=['ID'])

    print("First clustering step completed.")
    print("Starting manual clusters assembling procedure.")

    are_clusters_ok = "n"

    #    unique_labels = set(labels_2)
    clusters_list = np.arange(n_clusters_)
    clusters_list_2 = clusters_list[:]

    while are_clusters_ok == "n":  # if n in clusters_list_2:

        # Black removed and is used for noise instead.
        unique_labels = set(labels_2)
        # clusters_list = np.arange(n_clusters_)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]

        for k, col in zip(unique_labels, colors):
            # for k, col in zip(clusters_list_2, colors):

            markersize = 3
            fontsize = 15

            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
                markersize = 0
                fontsize = 0

            class_member_mask = (labels == k)

            xy = feature_matrix[class_member_mask & core_samples_mask]
            # np.delete(xy, 2, 1)

            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markeredgewidth=0.0,
                     markersize=markersize)
            #            x = xy[:, 0]
            #            y = xy[:, 1]

            for i, ((x, y, z),) in enumerate(zip(xy)):
                plt.text(x, y, str(k), ha="center", va="center", fontsize=fontsize, color='green')

                if i == 0:
                    break

            xy = feature_matrix[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markeredgewidth=0.0, markersize=markersize)

        plt.show()
        plt.clf()

        s()
        are_clusters_ok = str(input("Are clusters ok? (y/n): "))
        s()
        if are_clusters_ok == "y":
            break

        print("Clusters:")
        print(clusters_list_2)
        s()
        merge1 = int(input("Which base cluster you want to merge: "))
        merge2_in = input(
            "Which clusters you want to merge into the cluster number " + str(merge1) + " (Example: 1,13,25): ")
        merge2_clean = merge2_in.split(",")
        merge2 = []
        for i in merge2_clean:
            merge2.append(int(i))
        s()
        for i in merge2:
            clusters_list_2 = np.delete(clusters_list_2, np.where(clusters_list_2 == i))

            labels_2[labels_2 == i] = merge1
            # gdf_clusters['clusters'].replace(clusters_list[n], merge)

    gdf_clusters['clusters'] = labels_2
    gdf_pop_noise = gdf_clusters[gdf_clusters['clusters'] == -1]
    noise_people = round(sum(gdf_pop_noise['Population']), 0)
    clustered_people = round(1 - noise_people / total_people, 2)
    clusters_list = np.vstack([clusters_list_2, clusters_list_2, clusters_list_2])
    # pop_load = int(input("What is the load estimation in kW per person?"))
    pop_load = 0.4  # estimation of 2kW per person
    for i in clusters_list_2:
        pop_cluster = gdf_clusters[gdf_clusters['clusters'] == i]
        clusters_list[1, np.where(clusters_list_2 == i)] = sum(pop_cluster['Population'])
        clusters_list[2, np.where(clusters_list_2 == i)] = clusters_list[1, np.where(clusters_list_2 == i)] * pop_load
    del gdf_pop_noise
    os.chdir(r'Output//Clusters')
    gdf_cluster_total = gdf_clusters[gdf_clusters['clusters'] != -1]
    gdf_clusters.to_file("gdf_clusters.shp")
    gdf_cluster_total.to_file("total_cluster.shp")
    print("Clustering completed and cluster file extracted")
    os.chdir(r'..//..')
    l()
    return gdf_clusters, clusters_list_2, clusters_list


def nearest(row, geom_union, df1, df2, geom1_col='geometry', geom2_col='geometry', src_column=None):
    """Find the nearest point and return the corresponding value from specified column."""
    # Find the geometry that is closest
    nearest = df2[geom2_col] == nearest_points(row[geom1_col], geom_union)[1]
    # Get the corresponding value from df2 (matching is based on the geometry)
    value = df2[nearest][src_column].get_values()[0]
    return value


def grid(gdf_clusters, geodf_in, Proj_coords, clusters_list_2, resolution, clusters_load):
    print("Starting grid creation")
    s()
    # pop_threshold = int(input("What's the population threshold for the grid algorithm?: "))
    pop_threshold = 1
    s()
    # unitary cost of electric line: dollars per kilometers
    #paycheck = float(input("Which is the unitary money expenditure for the realization of a kilometer of grid?: "))
    paycheck = 10000
    s()
    limitHV = 3000
    limitMV = 300

    grid_merged = connection_merged = pd.DataFrame()
    grid_resume = pd.DataFrame(index=clusters_list_2,
                               columns=['Cluster', 'Grid_Length', 'Grid_Cost', 'Connection_Length',
                                        'Connection_Cost', 'ClusterLoad', 'ConnectionType'])
    grid_resume['Cluster'] = clusters_list_2
    grid_resume['ClusterLoad'] = clusters_load[2]
    os.chdir(r'Input//0_Dataset//Substations')
    substations = gpd.read_file("substations.shp")
    substations['ID'] = range(0, len(substations))
    os.chdir(r'..//..//..')
    os.chdir(r'Output//Grids')
    for cluster_n in clusters_list_2:

        #  SUBSTATIONS -----------------------------------------------------------------
        substations_new = substations
        if clusters_load[2][np.where(clusters_list_2 == cluster_n)[0][0]] > limitHV:
            substations_new = substations[substations['Type'] == 'HV']
        elif clusters_load[2][np.where(clusters_list_2 == cluster_n)[0][0]] > limitMV and clusters_load[2][
            np.where(clusters_list_2 == cluster_n)[0][0]] < limitHV:
            substations_new = substations[substations['Type'] == 'MV_3P']
        elif clusters_load[2][np.where(clusters_list_2 == cluster_n)[0][0]] < limitMV:
            substations_new = substations[substations['Type'] != 'HV']

        unary_union = geodf_in.unary_union
        substations_new['nearest_id'] = substations_new.apply(nearest, geom_union=unary_union, df1=substations_new, df2=geodf_in,
                                                      geom1_col='geometry', src_column='ID', axis=1)
        subinmesh = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
        for i, row in substations_new.iterrows():
            subinmesh = subinmesh.append(geodf_in[geodf_in['ID'] == row['nearest_id']], sort=False)
        subinmesh.reset_index(drop=True, inplace=True)
        # -------------------------------------------------------------------------------
        gdf_cluster = gdf_clusters[gdf_clusters['clusters'] == cluster_n]
        gdf_cluster_pop = gdf_cluster[gdf_cluster['Population'] >= pop_threshold]
        print(gdf_cluster_pop)

        # ---------------
        d1 = {'x': gdf_cluster_pop['X'], 'y': gdf_cluster_pop['Y'], 'z': gdf_cluster_pop['Elevation']}
        cluster_loc = pd.DataFrame(data=d1)
        cluster_loc.index = gdf_cluster_pop['ID']

        d2 = {'x': subinmesh['X'], 'y': subinmesh['Y'], 'z': subinmesh['Elevation']}
        sub_loc = pd.DataFrame(data=d2)
        sub_loc.index = subinmesh['ID']
        Distance_3D = pd.DataFrame(cdist(cluster_loc.values, sub_loc.values, 'euclidean'), index=cluster_loc.index,
                                   columns=sub_loc.index)
        mm = Distance_3D.min().idxmin()
        substation_designata = subinmesh[subinmesh['ID'] == mm]
        connectiontype = substations_new[substations_new['nearest_id']==mm]
        grid_resume['ConnectionType'][cluster_n] = connectiontype.Type.values[0]
        l()
        print("Creating grid for cluster n." + str(cluster_n) + " of " + str(len(clusters_list_2)))
        points_to_electrify = int(len(gdf_cluster_pop))
        l()
        if points_to_electrify == 0:
            ombelico_length = 0
            ombelico_cost = 0
            cluster_cost = 0
            cluster_line_length = 0

        elif points_to_electrify < 3:
            cluster_grid = gdf_cluster_pop
            fileout = 'Grid_' + str(cluster_n) + '.shp'

            cluster_grid.to_file(fileout)
            direct_connection, direct_cost, direct_length = grid_direct_connection(geodf_in, gdf_cluster_pop,
                                                                                   substation_designata,
                                                                                   Proj_coords, paycheck, resolution)
            direct = 'Connection_' + str(cluster_n) + '.shp'

            direct_connection.to_file(direct)
            ombelico_length = direct_length
            ombelico_cost = direct_cost
            cluster_cost = 0
            cluster_line_length = 0
            connection_merged = gpd.GeoDataFrame(pd.concat([connection_merged, direct_connection], sort=True))


        elif points_to_electrify < 400:
            cluster_grid1, cluster_cost1, cluster_line_length1, connections1 = Steinerman(geodf_in, gdf_cluster_pop,
                                                                                          Proj_coords, paycheck,resolution)
            cluster_grid2, cluster_cost2, cluster_line_length2, connections2 = Spiderman(geodf_in, gdf_cluster_pop,
                                                                                         Proj_coords, paycheck,
                                                                                         resolution)

            if cluster_cost1 <= cluster_cost2:
                print(" Steiner Wins!")
                cluster_line_length = cluster_line_length1
                cluster_cost = cluster_cost1
                fileout = 'Grid_' + str(cluster_n) + '.shp'
                cluster_grid1.to_file(fileout)
                grid_merged = gpd.GeoDataFrame(pd.concat([grid_merged, cluster_grid1], sort=True))

                checkpoint = list(cluster_grid1['ID1'].values) + list(cluster_grid1['ID2'].values)
                checkpoint = list(dict.fromkeys(checkpoint))
                checkpoint_ext = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
                for i in checkpoint:
                    checkpoint_ext = checkpoint_ext.append(geodf_in[geodf_in['ID'] == i], sort=False)
                checkpoint_ext.reset_index(drop=True, inplace=True)
                unary_union = checkpoint_ext.unary_union
                substation_designata['nearest_gridpoint'] = substation_designata.apply(nearest, geom_union=unary_union,
                                                                                       df1=substation_designata,
                                                                                       df2=checkpoint_ext,
                                                                                       geom1_col='geometry',
                                                                                       src_column='ID', axis=1)
                # -------------------------
                Pointsub = Point(substation_designata['X'], substation_designata['Y'])
                idgrid = int(substation_designata['nearest_gridpoint'].values)
                Pointgrid = Point(float(geodf_in[geodf_in['ID'] == idgrid].X),
                                  float(geodf_in[geodf_in['ID'] == idgrid].Y))
                dist = Pointsub.distance(Pointgrid)
                if dist < 30000:
                    cordone_ombelicale, ombelico_cost, ombelico_length = grid_connection(geodf_in, substation_designata,
                                                                                         Proj_coords, connections1,
                                                                                         paycheck, resolution)

                    cordone = 'connection_' + str(cluster_n) + '.shp'
                    cordone_ombelicale.to_file(cordone)
                    if cordone_ombelicale.type.values[0] == 'LineString':
                        connection_merged = gpd.GeoDataFrame(pd.concat([connection_merged, cordone_ombelicale], sort=True))
                else:
                    print('Cluster too far from the main grid to be connected, a microgrid solution is suggested')
                    ombelico_cost = 0
                    ombelico_length = 99999

            else:
                print("Steiner Loses")
                cluster_line_length = cluster_line_length2
                cluster_cost = cluster_cost2
                fileout = "Grid_" + str(cluster_n) + '.shp'
                cluster_grid2.to_file(fileout)
                grid_merged = gpd.GeoDataFrame(pd.concat([grid_merged, cluster_grid2], sort=True))

                checkpoint = list(cluster_grid2['ID1'].values) + list(cluster_grid2['ID2'].values)
                checkpoint = list(dict.fromkeys(checkpoint))
                checkpoint_ext = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
                for i in checkpoint:
                    checkpoint_ext = checkpoint_ext.append(geodf_in[geodf_in['ID'] == i], sort=False)
                checkpoint_ext.reset_index(drop=True, inplace=True)
                unary_union = checkpoint_ext.unary_union
                substation_designata['nearest_gridpoint'] = substation_designata.apply(nearest, geom_union=unary_union,
                                                                                       df1=substation_designata,
                                                                                       df2=checkpoint_ext,
                                                                                       geom1_col='geometry',
                                                                                       src_column='ID', axis=1)
                # -------------------------
                Pointsub = Point(substation_designata['X'], substation_designata['Y'])
                idgrid = int(substation_designata['nearest_gridpoint'].values)
                Pointgrid = Point(float(geodf_in[geodf_in['ID'] == idgrid].X),
                                  float(geodf_in[geodf_in['ID'] == idgrid].Y))
                dist = Pointsub.distance(Pointgrid)
                if dist < 30000:
                    cordone_ombelicale, ombelico_cost, ombelico_length = grid_connection(geodf_in, substation_designata,
                                                                                         Proj_coords, connections2,
                                                                                         paycheck, resolution)

                    cordone = 'connection_' + str(cluster_n) + '.shp'
                    cordone_ombelicale.to_file(cordone)
                    if cordone_ombelicale.type.values[0] == 'LineString':
                        connection_merged = gpd.GeoDataFrame(pd.concat([connection_merged, cordone_ombelicale], sort=True))
                else:
                    print('Cluster too far from the main grid to be connected, a microgrid solution is suggested')
                    ombelico_cost = 0
                    ombelico_length = 99999
        else:
            print('Too many points to electrify, Steiner not recommended.')
            cluster_grid2, cluster_cost, cluster_line_length, connections = Spiderman(geodf_in, gdf_cluster_pop,
                                                                                      Proj_coords, paycheck, resolution)

            fileout = "Grid_" + str(cluster_n) + '.shp'
            cluster_grid2.to_file(fileout)
            grid_merged = gpd.GeoDataFrame(pd.concat([grid_merged, cluster_grid2], sort=True))

            checkpoint = list(cluster_grid2['ID1'].values) + list(cluster_grid2['ID2'].values)
            checkpoint = list(dict.fromkeys(checkpoint))
            checkpoint_ext = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
            for i in checkpoint:
                checkpoint_ext = checkpoint_ext.append(geodf_in[geodf_in['ID'] == i], sort=False)
            checkpoint_ext.reset_index(drop=True, inplace=True)
            unary_union = checkpoint_ext.unary_union
            substation_designata['nearest_gridpoint'] = substation_designata.apply(nearest, geom_union=unary_union,
                                                                                   df1=substation_designata,
                                                                                   df2=checkpoint_ext,
                                                                                   geom1_col='geometry',
                                                                                   src_column='ID', axis=1)
            # -------------------------
            Pointsub = Point(substation_designata['X'], substation_designata['Y'])
            idgrid = int(substation_designata['nearest_gridpoint'].values)
            Pointgrid = Point(float(geodf_in[geodf_in['ID'] == idgrid].X),
                              float(geodf_in[geodf_in['ID'] == idgrid].Y))
            dist = Pointsub.distance(Pointgrid)
            if dist < 30000:
                cordone_ombelicale, ombelico_cost, ombelico_length = grid_connection(geodf_in, substation_designata,
                                                                                     Proj_coords, connections,
                                                                                     paycheck, resolution)
                if dist != 0:
                    cordone = 'connection_' + str(cluster_n) + '.shp'
                    cordone_ombelicale.to_file(cordone)
                    if cordone_ombelicale.type.values[0] == 'LineString':
                        connection_merged = gpd.GeoDataFrame(pd.concat([connection_merged, cordone_ombelicale], sort=True))
            else:
                print('Cluster too far from the main grid to be connected, a microgrid solution is suggested')
                ombelico_cost = 0
                ombelico_length = 9999999999

        grid_resume.at[cluster_n, 'Grid_Length'] = cluster_line_length/1000
        grid_resume.at[cluster_n, 'Grid_Cost'] = cluster_cost/1000
        grid_resume.at[cluster_n, 'Connection_Length'] = ombelico_length/1000
        grid_resume.at[cluster_n, 'Connection_Cost'] = ombelico_cost/1000


    grid_resume.to_csv('grid_resume.csv')
    connection_merged.crs = grid_merged.crs = geodf_in.crs
    connection_merged.to_file('merged_connections')
    grid_merged.to_file('merged_grid')
    os.chdir(r'..//..')
    print("All grids have been optimized and exported.")
    l()
    return grid_resume, paycheck


def load():
    print("8.Load creation")
    s()
    peop_per_house = int(input("How many people are there on average in an household?: "))
    s()

    os.chdir(r'Output//Clusters')
    clusters = gpd.read_file('total_cluster.shp')
    n_clusters = max(clusters['clusters']) + 1
    os.chdir(r'..//..')

    os.chdir(r'Input//3_load_ref')
    clusters_list_2 = list(set(clusters['clusters']))

    load_ref = pd.read_csv('Load.csv', index_col=0)
    users_ref = load_ref.loc[['Numero di utenti per categoria']]
    house_ref = int(load_ref['Case "premium"']['Numero di utenti per categoria']
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

        public_services = ["Clinica", "Scuola Primaria", "Scuola Secondaria", "Amministrazione"]

        for service in load_cluster.columns.values:

            users_ref = load_ref.loc[['Numero di utenti per categoria']]

            if service in public_services:
                load_cluster.at['Numero di utenti per categoria', service] = 1

            else:

                new_n_users = round(
                    users_ref[service]['Numero di utenti per categoria'] / house_ref * cluster_houses + 0.5, 0)
                load_cluster.at['Numero di utenti per categoria', service] = new_n_users

        for i in hours:
            for j in load_cluster.columns.values:
                new_load = round(load_ref[j][i] / house_ref * cluster_houses, 3)
                load_cluster.at[i, j] = new_load

                users_ref = load_cluster.loc[['Numero di utenti per categoria']]  # NON CANCELLARE

        name_load = "load_cluster_n_" + str(n) + '_houses_' + str(cluster_houses) + '.csv'
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


def grid_connection(mesh, substation_designata, Proj_coords, connections, paycheck, resolution):
    Pointsub = Point(substation_designata['X'], substation_designata['Y'])
    idsub = int(substation_designata['ID'].values)
    idgrid = int(substation_designata['nearest_gridpoint'].values)
    Pointgrid = Point(float(mesh[mesh['ID'] == idgrid].X), float(mesh[mesh['ID'] == idgrid].Y))
    dist = Pointsub.distance(Pointgrid)
    if dist < 1000:
        extension = dist
    elif dist < 2000:
        extension = dist / 2
    else:
        extension = dist / 4

    xmin = min(Pointsub.x, Pointgrid.x)
    xmax = max(Pointsub.x, Pointgrid.x)
    ymin = min(Pointsub.y, Pointgrid.y)
    ymax = max(Pointsub.y, Pointgrid.y)
    bubble = box(minx=xmin - extension, maxx=xmax + extension, miny=ymin - extension, maxy=ymax + extension)
    df_box = mesh[mesh.within(bubble)]
    df_box.index = pd.Series(range(0, len(df_box['ID'])))

    solocoord2 = {'x': df_box['X'], 'y': df_box['Y']}
    solocoord3 = {'x': df_box['X'], 'y': df_box['Y'], 'z': df_box['Elevation']}
    Battlefield2 = pd.DataFrame(data=solocoord2)
    Battlefield2.index = df_box['ID']
    Battlefield3 = pd.DataFrame(data=solocoord3)
    Battlefield3.index = df_box['ID']
    print('Checkpoint 2: Connecting to the nearest Substation')

    Distance_2D_box = distance_matrix(Battlefield2, Battlefield2)
    if np.any(Distance_2D_box):

        Distance_3D_box = pd.DataFrame(cdist(Battlefield3.values, Battlefield3.values, 'euclidean'),
                                       index=Battlefield3.index, columns=Battlefield3.index)
        Weight_matrix = weight_matrix(df_box, Distance_3D_box, paycheck)
        print('3D distance matrix has been created')
        # min_length = Distance_2D_box[Distance_2D_box > 0].min()
        # if min_length < 0.7*resolution:
        #     diag_length = min_length * math.sqrt(2)
        # elif min_length > 0.5*resolution:
        diag_length = resolution * 1.5
        # Definition of weighted connections matrix
        Edges_matrix = Weight_matrix
        # Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 999999999
        Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 0

        print('Checkpoint 4: Edges matrix completed')
        for i in connections:
            if i[0] in Edges_matrix.index.values and i[1] in Edges_matrix.index.values:
                # print(i[0], i[1])
                Edges_matrix.loc[i[0], i[1]] = 0.01
                Edges_matrix.loc[i[1], i[0]] = 0.01

        Edges_matrix_sparse = sparse.csr_matrix(Edges_matrix)
        Graph = nx.from_scipy_sparse_matrix(Edges_matrix_sparse)
        #    Graph.remove_edges_from(np.where(Edges_matrix.to_numpy() == 999999999))
        source = df_box.loc[df_box['ID'].values == idsub, :]
        target = df_box.loc[df_box['ID'].values == idgrid, :]
        source = source.index[0]
        target = target.index[0]
        print('Checkpoint 5')
        # Lancio Dijkstra e ottengo il percorso ottimale per connettere 'source' a 'target'
        path = nx.dijkstra_path(Graph, source, target, weight='weight')
        Steps = len(path)
        # Analisi del path
        r = 0  # Giusto un contatore
        PathID = []
        Digievoluzione = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
        Digievoluzione['id'] = pd.Series(range(1, Steps))
        Digievoluzione['x1'] = pd.Series(range(1, Steps))
        Digievoluzione['x2'] = pd.Series(range(1, Steps))
        Digievoluzione['y1'] = pd.Series(range(1, Steps))
        Digievoluzione['y2'] = pd.Series(range(1, Steps))
        Digievoluzione['ID1'] = pd.Series(range(1, Steps))
        Digievoluzione['ID2'] = pd.Series(range(1, Steps))
        Digievoluzione['Weight'] = pd.Series(range(1, Steps))
        Digievoluzione['geometry'] = None
        Digievoluzione['geometry'].stype = gpd.geoseries.GeoSeries
        print('Checkpoint 6:')

        for h in range(0, Steps - 1):
            con = [min(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']),
                   max(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID'])]
            if con not in connections:
                PathID.append(df_box.loc[path[h], 'ID'])
                PathID.append(df_box.loc[path[h + 1], 'ID'])
                Digievoluzione.at[r, 'geometry'] = LineString(
                    [(df_box.loc[path[h], 'X'], df_box.loc[path[h], 'Y'], df_box.loc[path[h], 'Elevation']),
                     (
                         df_box.loc[path[h + 1], 'X'], df_box.loc[path[h + 1], 'Y'],
                         df_box.loc[path[h + 1], 'Elevation'])])
                Digievoluzione.at[r, ['id']] = 0
                Digievoluzione.at[r, ['x1']] = df_box.loc[path[h], 'X']
                Digievoluzione.at[r, ['x2']] = df_box.loc[path[h + 1], 'X']
                Digievoluzione.at[r, ['y1']] = df_box.loc[path[h], ['Y']]
                Digievoluzione.at[r, ['y2']] = df_box.loc[path[h + 1], ['Y']]
                Digievoluzione.at[r, 'ID1'] = df_box.loc[path[h], 'ID']
                Digievoluzione.at[r, 'ID2'] = df_box.loc[path[h + 1], 'ID']
                Digievoluzione.at[r, 'Weight'] = Edges_matrix.loc[
                    df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']]
                connections.append(con)
                r += 1
        print('Checkpoint 7: Genomica di digievoluzione identificata')
        # Elimino le righe inutili di Digievoluzione e i doppioni da PathID
        indexNames = Digievoluzione[Digievoluzione['id'] > 0].index
        Digievoluzione.drop(indexNames, inplace=True)
        PathID = list(dict.fromkeys(PathID))
        cordone_ombelicale = Digievoluzione
        cordone_ombelicale = cordone_ombelicale.drop(['id', 'x1', 'x2', 'y1', 'y2'], axis=1)
        cordone_ombelicale['line_length'] = cordone_ombelicale['geometry'].length
        total_cost = int(cordone_ombelicale['Weight'].sum(axis=0))
        total_length = cordone_ombelicale['line_length'].sum(axis=0)

        return cordone_ombelicale, total_cost, total_length
    elif not np.any(Distance_2D_box):
        cordone_ombelicale = substation_designata
        total_cost = 0
        total_length = 0
        return cordone_ombelicale, total_cost, total_length


def grid_direct_connection(mesh, gdf_cluster_pop, substation_designata, Proj_coords, paycheck, resolution):
    print('Only few points to electrify, no internal cluster grid necessary. Connecting to the nearest substation')
    Pointsub = Point(substation_designata['X'], substation_designata['Y'])
    idsub = int(substation_designata['ID'].values)
    if len(gdf_cluster_pop.ID) == 1:
        Pointgrid = Point(float(gdf_cluster_pop['X']), float(gdf_cluster_pop['Y']))
        idgrid = int(gdf_cluster_pop['ID'].values)
        dist = Pointsub.distance(Pointgrid)
    elif len(gdf_cluster_pop.ID) > 0:  # checks if there's more than 1 point, connected the closest one
        d_point = {}
        d_dist = {}
        for i in range(0, len(gdf_cluster_pop.ID)):
            d_point[i] = Point(float(gdf_cluster_pop.X.values[i]), float(gdf_cluster_pop.Y.values[i]))
            d_dist[i] = Pointsub.distance(d_point[i])
        if d_dist[1] > d_dist[0]:
            Pointgrid = d_point[0]
            dist = d_dist[0]
            idgrid = int(gdf_cluster_pop.ID.values[0])
        else:
            Pointgrid = d_point[1]
            dist = d_dist[1]
            idgrid = int(gdf_cluster_pop.ID.values[1])
    if dist < 1000:
        extension = dist
    elif dist < 2000:
        extension = dist / 2
    else:
        extension = dist / 4

    xmin = min(Pointsub.x, Pointgrid.x)
    xmax = max(Pointsub.x, Pointgrid.x)
    ymin = min(Pointsub.y, Pointgrid.y)
    ymax = max(Pointsub.y, Pointgrid.y)
    bubble = box(minx=xmin - extension, maxx=xmax + extension, miny=ymin - extension, maxy=ymax + extension)
    df_box = mesh[mesh.within(bubble)]
    df_box.index = pd.Series(range(0, len(df_box['ID'])))

    solocoord2 = {'x': df_box['X'], 'y': df_box['Y']}
    solocoord3 = {'x': df_box['X'], 'y': df_box['Y'], 'z': df_box['Elevation']}
    Battlefield2 = pd.DataFrame(data=solocoord2)
    Battlefield2.index = df_box['ID']
    Battlefield3 = pd.DataFrame(data=solocoord3)
    Battlefield3.index = df_box['ID']
    print('Checkpoint 2')

    Distance_2D_box = distance_matrix(Battlefield2, Battlefield2)
    Distance_3D_box = pd.DataFrame(cdist(Battlefield3.values, Battlefield3.values, 'euclidean'),
                                   index=Battlefield3.index, columns=Battlefield3.index)
    Weight_matrix = weight_matrix(df_box, Distance_3D_box, paycheck)
    print('3D distance matrix has been created')
    min_length = Distance_2D_box[Distance_2D_box > 0].min()
    diag_length = resolution * 1.5

    # Definition of weighted connections matrix
    Edges_matrix = Weight_matrix
    # Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 999999999
    Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 0
    Edges_matrix_sparse = sparse.csr_matrix(Edges_matrix)
    print('Checkpoint 4: Edges matrix completed')
    # Graph = nx.from_numpy_matrix(Edges_matrix.to_numpy())
    Graph = nx.from_scipy_sparse_matrix(Edges_matrix_sparse)
    #    Graph.remove_edges_from(np.where(Edges_matrix.to_numpy() == 999999999))
    source = df_box.loc[df_box['ID'].values == idsub, :]
    target = df_box.loc[df_box['ID'].values == idgrid, :]
    source = source.index[0]
    target = target.index[0]
    print('Checkpoint 5')
    # Lancio Dijkstra e ottengo il percorso ottimale per connettere 'source' a 'target'
    path = nx.dijkstra_path(Graph, source, target, weight='weight')
    Steps = len(path)
    # Analisi del path
    r = 0  # Giusto un contatore
    PathID = []
    Digievoluzione = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
    Digievoluzione['id'] = pd.Series(range(1, Steps))
    Digievoluzione['x1'] = pd.Series(range(1, Steps))
    Digievoluzione['x2'] = pd.Series(range(1, Steps))
    Digievoluzione['y1'] = pd.Series(range(1, Steps))
    Digievoluzione['y2'] = pd.Series(range(1, Steps))
    Digievoluzione['ID1'] = pd.Series(range(1, Steps))
    Digievoluzione['ID2'] = pd.Series(range(1, Steps))
    Digievoluzione['Weight'] = pd.Series(range(1, Steps))
    Digievoluzione['geometry'] = None
    Digievoluzione['geometry'].stype = gpd.geoseries.GeoSeries
    print('Checkpoint 6:')

    for h in range(0, Steps - 1):
        con = [min(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']),
               max(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID'])]
        # if con not in connections:
        PathID.append(df_box.loc[path[h], 'ID'])
        PathID.append(df_box.loc[path[h + 1], 'ID'])
        Digievoluzione.at[r, 'geometry'] = LineString(
            [(df_box.loc[path[h], 'X'], df_box.loc[path[h], 'Y'], df_box.loc[path[h], 'Elevation']),
             (
                 df_box.loc[path[h + 1], 'X'], df_box.loc[path[h + 1], 'Y'],
                 df_box.loc[path[h + 1], 'Elevation'])])
        Digievoluzione.at[r, ['id']] = 0
        Digievoluzione.at[r, ['x1']] = df_box.loc[path[h], 'X']
        Digievoluzione.at[r, ['x2']] = df_box.loc[path[h + 1], 'X']
        Digievoluzione.at[r, ['y1']] = df_box.loc[path[h], ['Y']]
        Digievoluzione.at[r, ['y2']] = df_box.loc[path[h + 1], ['Y']]
        Digievoluzione.at[r, 'ID1'] = df_box.loc[path[h], 'ID']
        Digievoluzione.at[r, 'ID2'] = df_box.loc[path[h + 1], 'ID']
        Digievoluzione.at[r, 'Weight'] = Edges_matrix.loc[
            df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']]
        # connections.append(con)
        r += 1

    print('Checkpoint 7: Genomica di digievoluzione identificata')
    # Elimino le righe inutili di Digievoluzione e i doppioni da PathID
    indexNames = Digievoluzione[Digievoluzione['id'] > 0].index
    Digievoluzione.drop(indexNames, inplace=True)
    PathID = list(dict.fromkeys(PathID))
    cordone_ombelicale = Digievoluzione
    cordone_ombelicale = cordone_ombelicale.drop(['id', 'x1', 'x2', 'y1', 'y2'], axis=1)
    cordone_ombelicale['line_length'] = cordone_ombelicale['geometry'].length
    total_cost = int(cordone_ombelicale['Weight'].sum(axis=0))
    total_length = cordone_ombelicale['line_length'].sum(axis=0)

    return cordone_ombelicale, total_cost, total_length


def sizing(loads_list, clusters_list_2, houses_list, house_ref):
    print("9.Generation dimensioning")
    print("Now you need to use an external tool in order to optimize the power generation block for each cluster.\n"
          "The energy consumption for each of them is in a CSV file named Output/Load/total_loads.csv")
    print("From our side we'll use the Calliope Energy Model in order to optimize the system. This will need you to\n"
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

    total_energy = pd.DataFrame(index=clusters_list_2, columns=['Total Energy Consumption [kWh]'])
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
        total_energy.at[c, 'Total Energy Consumption [kWh]'] = energy * project_life

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
        total_undiscounted_cost = calliope_x_GISEle(time_span, c, cost_km_line, project_life)
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
                               columns=['GRID_NPC [$]', 'MG_NPC [$]', 'Total Energy Consumption [kWh]',
                                        'MG_LCOE [$/kWh]', 'GRID_LCOE [$/kWh]', 'Suggested Solution'])

    print(total_energy)
    s()
    print(grid_resume)
    s()
    print(mg_NPC)
    s()

    for clu in clusters_list_2:
        print(clu)
        final_LCOEs.at[clu, 'Total Energy Consumption [kWh]'] = total_energy.loc[clu, 'Total Energy Consumption [kWh]']
        final_LCOEs.at[clu, 'GRID_NPC [$]'] = grid_resume.loc[clu, 'Grid_Cost'] + grid_resume.loc[
            clu, 'Connection_Cost']

        final_LCOEs.at[clu, 'MG_NPC [$]'] = mg_NPC.loc[clu, 'Total Cost [$]']

        mg_lcoe = final_LCOEs.loc[clu, 'MG_NPC [$]'] / final_LCOEs.loc[clu, 'Total Energy Consumption [kWh]']

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


def grid_optimization(gdf_clusters, geodf_in, grid_resume, proj_coords, resolution, paycheck):
    os.chdir('D:\Vinicius\Mestrado\Thesis\GISELE\Output\Grids')
    new_connection_merged = pd.DataFrame()
    grid_resume = pd.read_csv('grid_resume.csv')
    grid_resume.reset_index(inplace=True, drop=True)
    grid_resume = grid_resume.sort_values(by='Connection_Cost')
    clusters_list_2 = grid_resume['Cluster']
    check = np.zeros(len(clusters_list_2), dtype=bool)
    grid_resume = grid_resume.sort_values(by='Cluster')
    # check = np.zeros(len(clusters_list_2), dtype=bool)

    count = 0

    while not any(check):  # checks if all variables in check are True
        for i in clusters_list_2:
            betterconnection = False
            print('Evaluating if there is a better connection for cluster ' + str(i))
            k = np.where(grid_resume['Cluster'] == i)  # find the index of the cluster we are analysing
            dist = pd.DataFrame(index=clusters_list_2, columns=['Distance', 'NearestPoint', 'Cost'])
            for j in clusters_list_2:
                k2 = np.where(grid_resume['Cluster'] == j)  # index of the cluster we trying to connect
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
                        pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == id]], sort=True))
                last_id = grid1.loc[
                    grid1.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
                chunk_points = gpd.GeoDataFrame(
                    pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == last_id]], sort=True))
                uu_grid1 = chunk_points.unary_union

                grid2 = gpd.read_file("Grid_" + str(i) + ".shp")
                chunk_points = pd.DataFrame()
                for id in grid2.ID1:  # getting all the points inside the chunk to compute the nearest_points
                    chunk_points = gpd.GeoDataFrame(
                        pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == id]], sort=True))
                last_id = grid2.loc[
                    grid2.ID2.size - 1, 'ID2']  # this lines are necessary because the last point was not taken
                chunk_points = gpd.GeoDataFrame(
                    pd.concat([chunk_points, gdf_clusters[gdf_clusters['ID'] == last_id]], sort=True))
                uu_grid2 = chunk_points.unary_union
                del chunk_points

                dist.NearestPoint[j] = nearest_points(uu_grid1, uu_grid2)
                p1 = gdf_clusters[gdf_clusters['geometry'] == dist.NearestPoint[j][0]]
                p2 = gdf_clusters[gdf_clusters['geometry'] == dist.NearestPoint[j][1]]
                # if the distance between grids is higher than 1.2 times the length  former connection, skip
                if (dist.NearestPoint[j][0].distance(dist.NearestPoint[j][1]))/1000 > \
                        1.2*(grid_resume.at[k[0][0],'Connection_Length']):
                    dist.Distance[j] = 9999999
                    dist.Cost[j] = 99999999
                    continue
                blockPrint()
                direct_connection, direct_cost, direct_length = grid_direct_connection(geodf_in, p1, p2,
                                                                                       proj_coords,
                                                                                       paycheck,
                                                                                       resolution)
                enablePrint()

                dist.Distance[j] = direct_length
                if grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0] == 'HV':
                    direct_cost = direct_cost + 1000000
                dist.Cost[j] = direct_cost
                if grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0] == 'MV_3P':
                    if grid_resume.ClusterLoad.at[k[0][0]] + grid_resume.ClusterLoad.at[k2[0][0]] > 3000:
                        continue
                if grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0] == 'MV_1P':
                    if grid_resume.ClusterLoad.at[k[0][0]] + grid_resume.ClusterLoad.at[k2[0][0]] > 300:
                        continue
                if min(dist.Cost) == direct_cost:
                    if direct_cost / 1000 < grid_resume.Connection_Cost.values[k]:
                        if check[np.where(clusters_list_2 == j)]:
                            direct = 'NewConnection_' + str(i) + '.shp'
                            direct_connection.to_file(direct)
                            print('A new connection for Cluster ' + str(i) + ' was successfully created')
                            grid_resume.at[k[0][0], 'Connection_Length'] = direct_length / 1000
                            grid_resume.at[k[0][0], 'Connection_Cost'] = direct_cost / 1000
                            grid_resume.at[k[0][0], 'ConnectionType'] = \
                            grid_resume.ConnectionType.values[np.where(grid_resume['Cluster'] == j)][0]
                            betterconnection = True
                            best_connection = direct_connection
            check[count] = True
            count = count + 1
            if betterconnection:
                new_connection_merged = gpd.GeoDataFrame(pd.concat([new_connection_merged, best_connection], sort=True))
            elif not betterconnection:
                if grid_resume.Connection_Length.values[k] > 0:
                    best_connection = gpd.read_file('connection_' + str(i) + '.shp')
                    new_connection_merged = gpd.GeoDataFrame(pd.concat([new_connection_merged, best_connection], sort=True))

    grid_resume.to_csv('grid_resume_opt.csv')
    new_connection_merged.crs = geodf_in.crs
    new_connection_merged.to_file('merged_connections_opt')
    return grid_resume
