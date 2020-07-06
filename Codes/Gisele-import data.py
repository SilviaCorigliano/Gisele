import ee
import osmnx as ox
import geopandas as gpd
import requests
import json
import pandas as pd
from shapely.geometry import Point,LineString
import numpy as np

#Input data#
crs='EPSG:32737'

def download_url(url, save_path, chunk_size=128):
    """
    Download zip file from specified url and save it into a defined folder
    Note: check chunk_size parameter
    """
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def donwload_tif(area,crs,scale,image,out_path):
    '''
    Download data from Earth Engine
    :param area: GeoDataFrame with the polygon of interest area
    :param crs: str with crs of the project
    :param scale: int with pixel size in meters
    :param image: image from the wanted database in Earth Image
    :param out_path: str with output path
    :return:
    '''
    minx, miny, maxx, maxy = area.geometry.total_bounds
    path = image.getDownloadUrl({
        'scale': scale,
        'crs': crs,
        'region': [[minx, miny], [minx, maxy], [maxx, miny], [maxx, maxy]]
    })
    print(path)
    download_url(path, out_path)
    return

##### Import layer with region boundaries and extract its extent#####
area=gpd.read_file('Input/Namajavira_4326.shp')
minx, miny, maxx, maxy = area.geometry.total_bounds
###############################################
#######Download layers from Earth Engine#######
################################################

# Initialize the Earth Engine module.
ee.Initialize()

#########  DEM with 30m resolution SRTM #########
print('Elevation')
image = ee.Image("USGS/SRTMGL1_003")
out_path='Output\Elevation.zip'
scale=30
donwload_tif(area,crs,scale,image,out_path)

####### Land Cover from Copernicus#########
#Copernicus Global Land Cover Layers: CGLS-LC100 collection 2. 100m resolution. Ref year 2015
#download only image related to discrete classification (see in the documentation)
print('Land Cover')
collection = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V/Global").select('discrete_classification')
image = collection.sort('system:time_start', False)\
    .first()
out_path='Output\LandCover.zip'
scale=100
donwload_tif(area,crs,scale,image,out_path)

###### Population from worldpop #######
print('Population')
scale=100
image = ee.ImageCollection("WorldPop/GP/100m/pop").filter(ee.Filter.date('2015-01-01', '2015-12-31')).select('population')
image= image.reduce(ee.Reducer.median())

###### Population from JRC #######
#scale=250
# image=ee.ImageCollection("JRC/GHSL/P2016/POP_GPW_GLOBE_V1").sort('system:time_start', False)\
#     .first().select('population_count')
out_path = 'Output\Population.zip'
donwload_tif(area,crs,scale,image,out_path)

########protected areas from WDPA (as polygons)########
#path = ee.FeatureCollection("WCMC/WDPA/current/polygons").getDownloadURL('csv')

#how to save it as shp?

###############################################
############Download layers from OSM###########
################################################

#roads layer

G = ox.graph_from_bbox(miny, maxy, minx, maxx, network_type='drive')
ox.save_graph_shapefile(G, filepath='Output/streets') #resulting crs is epsg:4326

#transmission grid

overpass_url = "http://overpass-api.de/api/interpreter"

overpass_query = """
[out:json];
(
 way["power"="line"]("""+str(miny)+','+str(minx)+','+str(maxy)+','+str(maxx)+''');
);
out geom;
'''
response = requests.get(overpass_url,
                        params={'data': overpass_query})
data = response.json()
df = pd.json_normalize(data['elements'])
df['geometry_new']=df['geometry']
for i,row in df.iterrows():
    points=[]
    for j in row['geometry']:
        print(j)
        points.append(Point(j['lon'],j['lat']))
    df.loc[i,'geometry_new']=LineString(points)
geodf=gpd.GeoDataFrame(df[['id','tags.power','tags.voltage','geometry_new']],geometry='geometry_new')
geodf.crs='epsg:4326'
geodf.to_crs(crs)
geodf.to_file('Output/power/transmissiongrid.shp')

overpass_query = """
[out:json];
(node["power"="substation"]("""+str(miny)+','+str(minx)+','+str(maxy)+','+str(maxx)+''');
 way["power"="substation"]('''+str(miny)+','+str(minx)+','+str(maxy)+','+str(maxx)+''');
);
out center;
'''
response = requests.get(overpass_url,
                        params={'data': overpass_query})
data = response.json()
df = pd.json_normalize(data['elements'])

# Collect coords into list
coords = []
for element in data['elements']:
  if element['type'] == 'node':
    lon = element['lon']
    lat = element['lat']
    coords.append((lon, lat))
  elif 'center' in element:
    lon = element['center']['lon']
    lat = element['center']['lat']
    coords.append((lon, lat))
# Convert coordinates into numpy array
X = np.array(coords)
gdf = gpd.GeoDataFrame(
    df[['id','tags.voltage','tags.substation']], geometry=gpd.points_from_xy(X[:,0], X[:,1]))
gdf.crs='epsg:4326'
gdf.to_crs(crs)
gdf.to_file('Output/power/substations.shp')

a=1


