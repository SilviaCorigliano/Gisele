"""
Born in Weed May 22 2019

@author: nondarloavedere
"""

'----------------------------------------------------------------------------------------------------------------------'
print("Importing GISEle complementary package!")
import sys
# sys.path.insert(0, 'Codes') #necessary for native GISEle Packages
# native GISEle Packages
import supporting_GISEle2

# from supporting_GISEle_pt_2 import *
print("GISEle complementary package successfully imported!")
'----------------------------------------------------------------------------------------------------------------------'
supporting_GISEle2.s()
'----------------------------------------------------------------------------------------------------------------------'
supporting_GISEle2.l()
'Installing packages'
print("1.Installing packages - Let's check if you have all necessary packages!")

print("You have all the necessary packages!")

'----------------------------------------------------------------------------------------------------------------------'
supporting_GISEle2.l()
"2.Importing packages"
print("2.Importing packages - Let's import import all packages!")

# system packages
import os
import glob

print("All system packages imported.")

# third party packages
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from shapely.geometry import Point
from fiona.crs import from_epsg
import geopandas as gpd

print("All third party packages packages imported.")
supporting_GISEle2.l()
'----------------------------------------------------------------------------------------------------------------------'

'Introduction'
supporting_GISEle2.l()
print("Welcome to GISEle! The steps are:")
supporting_GISEle2.l()
steps = ["1.Installing packages -completed-",
         "2.Importing packages -completed-",
         "3.Importing and processing base layer",
         "4.Assigning weights, Cleaning and creating the DataFrame",
         "5.DataFrame preparation",
         "6.Clustering",
         "7.Grid creation",
         "8.Load creation",
         "9.Generation sizing",
         "10.Final results"]
print("\n".join(steps))
supporting_GISEle2.s()
step = int(input("Which step do you want to select?: "))
# step = 7
if step <= 5:

    "3. Importing and processing base layer"

    points, proj_coords, unit, file_name, resolution = supporting_GISEle2.import_csv_file()
    '------------------------------------------------------------------------------------------------------------------'
    "4. Assigning weights, Cleaning and creating the DataFrame"

    df_in = supporting_GISEle2.carwash(points)
    geo_DataFrame = supporting_GISEle2.creating_dataframe(Point, df_in, proj_coords, unit, file_name)
    '------------------------------------------------------------------------------------------------------------------'
    "5.DataFrame preparation"

    total_points, total_people = supporting_GISEle2.info_dataframe(geo_DataFrame)  # info on the dataframe
    pop_points = supporting_GISEle2.dataframe_3D(geo_DataFrame)  # elevation as third dimension

    '------------------------------------------------------------------------------------------------------------------'
    "6.Clustering"

    feature_matrix, pop_weight = supporting_GISEle2.clustering_sensitivity(pop_points, geo_DataFrame, total_points,
                                                                           total_people)
    gdf_clusters, clusters_list_2, clusters_load = supporting_GISEle2.clustering(feature_matrix, pop_weight, geo_DataFrame,
                                                                  total_people)

    '------------------------------------------------------------------------------------------------------------------'
    "7.Grid creation"

    grid_resume, paycheck = supporting_GISEle2.grid(gdf_clusters, geo_DataFrame, proj_coords, clusters_list_2,
                                                    resolution, clusters_load)
    '------------------------------------------------------------------------------------------------------------------'
    "8.Grid optimization"

    grid_optimized = supporting_GISEle2.grid_optimization(gdf_clusters, geo_DataFrame, grid_resume, proj_coords,
                                                          resolution, paycheck)
    '------------------------------------------------------------------------------------------------------------------'

    "9.Load creation"

    loads_list, houses_list, house_ref = supporting_GISEle2.load()
    '------------------------------------------------------------------------------------------------------------------'
    "10.Generation sizing"

    total_energy, mg_NPC = supporting_GISEle2.sizing(loads_list, clusters_list_2, houses_list, house_ref)

    '------------------------------------------------------------------------------------------------------------------'
    "11.Final results"
    final_LCOEs = supporting_GISEle2.final_results(clusters_list_2, total_energy, grid_resume, mg_NPC)

elif step == 6:

    "*.DataFrame preparation"

    geo_DataFrame, proj_coords, unit, file_name, resolution = supporting_GISEle2.import_weighted_file()
    print(geo_DataFrame)
    total_points, total_people = supporting_GISEle2.info_dataframe(geo_DataFrame)  # info on the dataframe
    pop_points = supporting_GISEle2.dataframe_3D(geo_DataFrame)  # elevation as third dimension

    '-----------------------------------------------------------------------------------------------------------------'
    "7.Clustering and Grid creation"

    feature_matrix, pop_weight = supporting_GISEle2.clustering_sensitivity(pop_points, geo_DataFrame, total_points,
                                                                           total_people)
    gdf_clusters, clusters_list_2, clusters_load = supporting_GISEle2.clustering(feature_matrix, pop_weight,
                                                                                geo_DataFrame, total_people)

    grid_resume, paycheck = supporting_GISEle2.grid(gdf_clusters, geo_DataFrame, proj_coords, clusters_list_2,
                                                    resolution, clusters_load)

    "8.Grid optimization"

    grid_optimized = supporting_GISEle2.grid_optimization(gdf_clusters, geo_DataFrame, grid_resume, proj_coords,
                                                          resolution, paycheck)


elif step == 7:

    "*.DataFrame preparation"

    geo_DataFrame, proj_coords, unit, file_name, resolution = supporting_GISEle2.import_weighted_file()
    print(geo_DataFrame)
    total_points, total_people = supporting_GISEle2.info_dataframe(geo_DataFrame)  # info on the dataframe
    pop_points = supporting_GISEle2.dataframe_3D(geo_DataFrame)  # elevation as third dimension

    '-----------------------------------------------------------------------------------------------------------------'
    "7. Grid creation"

    gdf_clusters, clusters_list_2, clusters_load = supporting_GISEle2.import_cluster()

    grid_resume, paycheck = supporting_GISEle2.grid(gdf_clusters, geo_DataFrame, proj_coords, clusters_list_2,
                                                     resolution, clusters_load)

    "8.Grid optimization"

    grid_optimized = supporting_GISEle2.grid_optimization(gdf_clusters, geo_DataFrame, grid_resume, proj_coords,
                                                          resolution, paycheck)


elif step == 9:

    "8.Load creation"

    loads_list, houses_list, house_ref = supporting_GISEle2.load()
    '------------------------------------------------------------------------------------------------------------------'
    "9.Generation sizing"

    clusters_list_2 = []
    total_energy, mg_NPC = supporting_GISEle2.sizing(loads_list, clusters_list_2, houses_list, house_ref)

    '------------------------------------------------------------------------------------------------------------------'
    "10.Final results"
    grid_resume = []

    final_LCOEs = supporting_GISEle2.final_results(clusters_list_2, total_energy, grid_resume, mg_NPC)
