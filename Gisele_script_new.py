import os

import base64
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import plotly.graph_objs as go
from shapely.geometry import Point
from functions import load, sizing
from gisele import initialization, clustering, processing, collecting, \
    optimization, results, grid, branches
import pyutilib.subprocess.GlobalData
from gisele import QGIS_processing_polygon as qgis_process

from gisele import MILP_Input_creation,MILP_models,process_output,grid_routing,Secondary_substations

gisele_folder=os.getcwd()
country  = 'Lesotho'
case_study='Thuso-10_clusters-180m_attempt'
final_clus=12
crs = 22287
resolution = 180
load_capita=0.7
pop_per_household=4
resolution_population = 30 # this should be automatic from the raster
landcover_option='ESACCI'
database= r'C:\Users\silvi\OneDrive - Politecnico di Milano\Documents\2020-2021\Gisele shared\8.Case_Study'

# This should be inside the procedure, only now we are taking the clusters as an input
cluster_folder = r'C:\Users\silvi\OneDrive - Politecnico di Milano\Documents\2020-2021\Gisele shared\8.Case_Study\Lesotho\Villages_areas - Copy.geojson'
substations_folder = r'C:\Users\silvi\OneDrive - Politecnico di Milano\Documents\2020-2021\Gisele shared\8.Case_Study\Lesotho\con_point - Copy.shp'
study_area = gpd.read_file(database+'/'+country+'/Study_area/Study_area.shp')
Clusters = gpd.read_file(cluster_folder)
Clusters = Clusters[Clusters['final_clus']==final_clus]
Clusters= Clusters.to_crs(crs)
Substations = gpd.read_file(substations_folder)
Substations = Substations[Substations['final_clus']==final_clus]
Substations_crs=Substations.to_crs(crs)
# this can actually be done before, or with the html file

Substations['X'] = [Substations_crs['geometry'].values[i].xy[0][0] for i in range(Substations.shape[0])]
Substations['Y'] = [Substations_crs['geometry'].values[i].xy[1][0] for i in range(Substations.shape[0])]
#Substations['Power'] = 20
#Substations['Cost'] = 0
#Substations['Distance']=0
#Substations['Voltage'] = 1
#Substations['ID'] = -Substations['OBJECTID']
if not os.path.exists(r'Case studies/'+case_study):
    os.makedirs(r'Case studies/'+case_study)
    os.makedirs(r'Case studies/'+case_study+'/Input')
    os.makedirs(r'Case studies/'+case_study+'/Output')

# specific data on the case study
LV_distance=500
ss_data = 'ss_data_Thuso.csv'
simplify_road_coef_inside = 5 # in meters, used for the routing inside the clusters.
simplify_road_coef_outside = 30 # in meters, used for creating connections among clusters/substations.
road_coef = 2

mg_option = False
reliability_option = True
Roads_option=True # Boolean stating whether we want to use the roads for the steiner tree and dijkstra.
n_line_type=1
# parameters for the cable that is used
resistance = 0.42
reactance = 0.39
Pmax=11
line_cost = 10000
# parameters of the grid that is being extended
voltage = 11 # kV

# parameters for the economical factors
coe = 150 # euro/MWh of electrical energy supplied
Substations.to_file(r'Case studies/'+case_study+'/Input/substations')
'''Create the grid of points'''
print('1. CREATE A WEIGHTED GRID OF POINTS')
#df_weighted = qgis_process.create_input_csv(crs,resolution,resolution_population,landcover_option,country,case_study,database,study_area)

'''For each cluster, perform further aglomerative clustering, locate secondary substations and perform MV grid routing'''
print('2. LOCATE SECONDARY SUBSTATIONS INSIDE THE CLUSTERS.')
# Secondary_substations.locate_secondary_ss(crs, resolution_population, load_capita, pop_per_household, road_coef,
#                        Clusters, case_study, LV_distance, ss_data,landcover_option, gisele_folder)
print('3. ROUTE THE MV GRID INSIDE THE CLUSTERS.')
grid_routing.routing(Clusters,gisele_folder,case_study,crs,resolution,Roads_option,simplify_road_coef_inside)
'''Create the input for the MILP'''
print('3. Create the input for the MILP.')
MILP_Input_creation.create_input(gisele_folder,case_study,crs,line_cost,resolution,mg_option,reliability_option,Roads_option,simplify_road_coef_outside)
'''Execute the desired MILP model'''
print('4. Execute the MILP according to the selected options.')
n_clusters = Clusters.shape[0]
if mg_option == False and reliability_option==True and n_line_type==1:
    MILP_models.MILP_without_MG(gisele_folder,case_study,n_clusters,coe,voltage,resistance,reactance,Pmax,line_cost)
'''Process the output from the MILP'''
print('5. Process MILP output')
process_output.process(gisele_folder,case_study)