import zipfile
import os
import rasterio.mask
from osgeo import gdal
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Point, MultiPoint, MultiPolygon
from shapely.ops import nearest_points
from shapely.geometry import Polygon
from rasterio.plot import show
from rasterio.mask import mask
import json
import matplotlib.pyplot as plt
import fiona
from collections import Counter
from statistics import mean
from math import ceil
from shapely import ops
from gisele import initialization
# Define the function resample
def resample(raster, resolution,options):

    # get pixel size
    affine = raster.transform
    pixel_x = affine[0]
    scale_factor = pixel_x / resolution
    # re-sample data to target shape
    if options == 'average':
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * scale_factor),
                int(raster.width * scale_factor)
            ),
            resampling=Resampling.average
        )
    elif options == 'mode':
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * scale_factor),
                int(raster.width * scale_factor)
            ),
            resampling=Resampling.mode
        )
    else:
        data = raster.read(
            out_shape=(
                raster.count,
                int(raster.height * scale_factor),
                int(raster.width * scale_factor)
            ),
            resampling=Resampling.bilinear
        )
    # scale image transform
    transform = raster.transform * raster.transform.scale(
        (raster.width / data.shape[-1]),
        (raster.height / data.shape[-2])
    )

    profile = raster.profile
    profile.update(transform=transform, width=raster.width * scale_factor,
                   height=raster.height * scale_factor)

    return data, profile

def create_grid(crs,resolution,study_area):
    # crs and resolution should be a numbers, while the study area is a polygon
    df = pd.DataFrame(columns=['X', 'Y'])
    min_x=float(study_area.bounds['minx'])
    min_y=float(study_area.bounds['miny'])
    max_x=float(study_area.bounds['maxx'])
    max_y = float(study_area.bounds['maxy'])
    # create one-dimensional arrays for x and y
    lon = np.arange(min_x, max_x, resolution)
    lat = np.arange(min_y, max_y, resolution)
    lon, lat = np.meshgrid(lon, lat)
    df['X'] = lon.reshape((np.prod(lon.shape),))
    df['Y'] = lat.reshape((np.prod(lat.shape),))
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y),
                            crs=crs)
    geo_df_clipped = gpd.clip(geo_df,study_area)
    #geo_df_clipped.to_file(r'Test\grid_of_points.shp')
    return geo_df_clipped

def rasters_to_points(study_area,crs,resolution_points,dir,protected_areas,streets,resolution_population):

    '''Read the point file'''
    pointData = create_grid(crs,resolution_points,study_area)
    #print(pointData.crs)
    pointData=pointData.to_crs(crs)
    '''Read all the shape files '''
    #change the resolution of the raster according to the grid
    population_Raster=rasterio.open(dir+'/Input/Population_' + str(crs) + '.tif')
    resolution_population = population_Raster.transform[0]
    data_population, profile_population = resample(population_Raster, resolution_points,'average')
    with rasterio.open(dir+'/Input/Population_resampled.tif',
                     'w', **profile_population) as dst:
        dst.write(data_population)
    Population = rasterio.open(dir+'/Input/Population_resampled.tif')
    ratio=Population.transform[0]/resolution_population

    elevation_Raster = rasterio.open(dir+'/Input/Elevation_' + str(crs) + '.tif')
    data_elevation, profile_elevation = resample(elevation_Raster, resolution_points,'bilinear')
    with rasterio.open(dir+'/Input/Elevation_resampled.tif',
                     'w', **profile_elevation) as dst:
        dst.write(data_elevation)
    #resolution_elevation = elevation_Raster.transform[0]
    Elevation = rasterio.open(dir+'/Input/Elevation_resampled.tif')

    slope_Raster = rasterio.open(dir+'/Input/Slope_' + str(crs) + '.tif')
    data_slope, profile_slope = resample(slope_Raster, resolution_points,'bilinear')
    with rasterio.open(dir+'/Input/Slope_resampled.tif',
                     'w', **profile_slope) as dst:
        dst.write(data_slope)

    Slope = rasterio.open(dir+'/Input/Slope_resampled.tif')

    landcover_Raster = rasterio.open(dir+'/Input/LandCover_' + str(crs) + '.tif')
    data_landcover, profile_landcover = resample(landcover_Raster, resolution_points,'mode')
    with rasterio.open(dir+'/Input/Landcover_resampled.tif',
                     'w', **profile_landcover) as dst:
        dst.write(data_landcover)

    LandCover = rasterio.open(dir+'/Input/Landcover_resampled.tif')
    resolution_landcover=landcover_Raster.transform[0]
    '''Create a dataframe for the final excel '''
    df = pd.DataFrame(columns=['ID', 'X', 'Y', 'Elevation', 'Slope',
                           'Population', 'Land_cover', 'Road_dist',
                           'River_flow', 'Protected_area'])
    print(pointData.shape)
    print(Elevation.transform)

    '''Easier way to do this procedure - find out who to do it for the roads '''
    coords = [(x, y) for x, y in zip(pointData.X, pointData.Y)]
    pointData = pointData.reset_index(drop=True)
    pointData['ID'] = pointData.index
    pointData['Elevation'] = [x[0] for x in Elevation.sample(coords)]
    print('Elevation finished')
    pointData['Slope'] = [x[0] for x in Slope.sample(coords)]
    print('Slope finished')
    pointData['Population'] = [x[0]*pow(ratio,2) for x in Population.sample(coords)]
    pointData['Population']=pointData['Population'].round(decimals=0)
    print('Population finished')
    pointData['Land_cover'] = [x[0] for x in LandCover.sample(coords)]
    print('Land cover finished')
    pointData['Protected_area'] = [ x for x in protected_areas['geometry'].contains(pointData.geometry)]
    print('Protected area finished')
    #nearest_geoms = nearest_points(pointData.geometry, streets)
    #pointData['Road_dist'] = [ x for x in nearest_geoms[0].distance(nearest_geoms[1])]
    #pointData.to_csv('Test/test.csv')
    '''Start iterating through the points to assign values from each raster '''
    road_distances=[]
    for index, point in pointData.iterrows():
        #print(point['geometry'])
        #print(point['geometry'].xy[0][0],point['geometry'].xy[1][0])
        x = point['geometry'].xy[0][0]
        y = point['geometry'].xy[1][0]
        '''Sample population in that exact same point(maybe we will need to find it as the average of the 9 points around'''
        #row, col = Population.index(x,y)
        #Pop=Population.read(1)[row,col]*pow(ratio,2) # because the population was resampled according to the average value, now we need to multiply

        #row, col= Elevation.index(x,y)
        #elevation = Elevation.read(1)[row,col]

        #row, col = Slope.index(x,y)
        #slope= Slope.read(1)[row,col]

        #row, col = LandCover.index(x,y)
        #landcover=LandCover.read(1)[row,col]

        #protected=protected_areas['geometry'].contains(Point(x,y)).any()

        nearest_geoms = nearest_points(Point(x,y), streets)
        road_distance = nearest_geoms[0].distance(nearest_geoms[1])
        road_distances.append(road_distance)
        #print(index)
        print('\r' + str(index) + '/' + str(pointData.index.__len__()),
              sep=' ', end='', flush=True)
        '''Add a point in the starting dataframe with all the information '''
        #df2 = pd.DataFrame([[int(index),x,y,elevation,slope,round(Pop),landcover,road_distance,0,protected]],
        #                    columns=['ID', 'X', 'Y', 'Elevation', 'Slope',
        #                        'Population', 'Land_cover', 'Road_dist',
        #                        'River_flow', 'Protected_area'])
        #df=df.append(df2)
    pointData['Road_dist']=road_distances
    pointData['River_flow'] = ""
    '''After the dataframe is created, change it to a geodataframe adding the geometry of each point '''
    #geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y),
                            #crs=crs)
    geo_df = gpd.GeoDataFrame(pointData, geometry=gpd.points_from_xy(pointData.X, pointData.Y),
                               crs=crs)
    print(pointData)
    print(geo_df)

    return pointData, geo_df


'''Set parameters for the analysis - by the user'''
crs=21095
resolution=500
resolution_population=30
country='Uganda'
case_study='Area in Uganda'
landcover_option='ESACCI'
'''Data processing'''
#database=r'Database/'+country
database = r'C:/Users/alekd/Politecnico di Milano/Silvia Corigliano - Gisele shared/8.Case_Study/'+country
crs_str=r'epsg:'+str(crs)
study_area = gpd.read_file(database+'/Study_area/Study_area.shp')
protected_areas = gpd.read_file(database+'/Protected_areas/Protected_area-Uganda.shp')
protected_areas = protected_areas.to_crs(crs)
streets = gpd.read_file(database+'/Roads/uga_trs_roads_osm.shp')
streets = streets.to_crs(crs)
files_present=False # set this to True if you already have all the .shp and raster files locally
study_area_crs=study_area.to_crs(crs)
dir=r'Case studies/'+case_study
if not os.path.exists(dir):
    os.makedirs(dir)
    os.makedirs(dir+'/Input')
    os.makedirs(dir + '/Output')
#ROADS
study_area_buffered=study_area.buffer((resolution*0.1/11250)/2)

study_area_buffered_list=[study_area_buffered] # this is to fix the issue with Polygon not being iterable when using rasterio
'''Clip the protected areas'''
protected_areas_clipped=gpd.clip(protected_areas,study_area_crs)
streets_clipped = gpd.clip(streets,study_area_crs)
if not files_present:
    if not streets_clipped.empty:
        streets_clipped.to_file(dir+'/Input/Roads.shp')

    if not protected_areas_clipped.empty:
        protected_areas_clipped.to_file(dir+'/Input/protected_area.shp')

    '''Clip the elevation and then change the crs'''
    with rasterio.open(database+'/Elevation/Elevation_new.tif',
            mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered, crop=True)
        print(src.crs)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir+'/Input/Elevation.tif', "w", **out_meta) as dest:
        dest.write(out_image)
    input_raster = gdal.Open(dir+'/Input/Elevation.tif')
    output_raster = dir+'/Input/Elevation_' + str(crs) + '.tif'
    warp = gdal.Warp(output_raster, input_raster, dstSRS=crs_str)
    warp = None  # Closes the files

    '''Clip the slope and then change the crs'''
    with rasterio.open(
            database+'/Slope/Slope_4326.tif',
            mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered, crop=True)
        print(src.crs)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir+'/Input/Slope.tif', "w", **out_meta) as dest:
        dest.write(out_image)
    input_raster = gdal.Open(dir+'/Input/Slope.tif')
    output_raster = dir+'/Input/Slope_' + str(crs) + '.tif'
    warp = gdal.Warp(output_raster, input_raster, dstSRS=crs_str)
    warp = None  # Closes the files

    '''Clip the population and then change the crs'''
    with rasterio.open(
            database+'/Population/Population-fb/hrsl_uga_zeros.tif',
            mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered, crop=True)
        print(src.crs)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir+'/Input/Population.tif', "w", **out_meta) as dest:
        dest.write(out_image)
    # for now there is no need since the population is already in the desired crs
    input_raster = gdal.Open(dir+'/Input/Population.tif')
    output_raster = dir+'/Input/Population_' + str(crs) + '.tif'
    warp = gdal.Warp(output_raster, input_raster, dstSRS=crs_str)
    warp = None  # Closes the files

    '''Clip the land cover and then change the crs'''
    with rasterio.open(
            database+'/LandCover/Uganda.tif',
            mode='r') as src:
        out_image, out_transform = rasterio.mask.mask(src, study_area_buffered, crop=True)
        print(src.crs)

    out_meta = src.meta
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(dir+'/Input/LandCover.tif', "w", **out_meta) as dest:
        dest.write(out_image)
    input_raster = gdal.Open(dir+'/Input/LandCover.tif')
    output_raster = dir+'/Input/LandCover_' + str(crs) + '.tif'
    warp = gdal.Warp(output_raster, input_raster, dstSRS=crs_str)
    warp = None  # Closes the files
'''This part transforms the roads from lines to multi points, we will see if it's needed'''
streets_points = []
for line in streets_clipped['geometry']:
    # print(line.geometryType)
    if line.geometryType() == 'MultiLineString':
        for line1 in line:
            for x in list(zip(line1.xy[0], line1.xy[1])):
                # print(line1)
                streets_points.append(x)
    else:
        for x in list(zip(line.xy[0], line.xy[1])):
            # print(line)
            streets_points.append(x)
streets_multipoint = MultiPoint(streets_points)

df, geo_df = rasters_to_points(study_area_crs, crs, resolution, dir,protected_areas_clipped,streets_multipoint,resolution_population)
geo_df.to_file(dir+'/Input/'+case_study+'.shp')
geo_df=geo_df.reset_index(drop=True)
geo_df['ID']=geo_df.index
df=df.reset_index(drop=True)
df['ID']=df.index
df.to_csv(dir+'/Input/'+case_study+'.csv', index=False)

'''cleaning the dataframe for easier clustering'''
df_clustering=df.drop(['Slope','Land_cover', 'Road_dist',
                        'River_flow', 'Protected_area','geometry'],axis=1)
df_clustering = df_clustering.drop(df_clustering[df_clustering.Population==0].index)
df_clustering.to_csv(dir+'/Input/'+case_study+'_clustering'+'.csv',index=False)

df_weighted= initialization.weighting(df, resolution, landcover_option)
df_weighted.to_csv(dir+'/Input/'+case_study+'_weighted.csv', index=False)