import zipfile
import os
from rasterio.warp import calculate_default_transform, reproject, Resampling #for reprojecting
import rasterio.mask
import fiona
from osgeo import gdal
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Point,LineString,MultiPoint
from shapely.ops import nearest_points

#Input data#
crs='EPSG:32737'
area=gpd.read_file('Input/Namajavira_4326.shp')
res=200
##### Reproject layer according to metric coordinate system#####
area=area.to_crs(crs)
minx, miny, maxx, maxy = area.geometry.total_bounds

###########create regular mesh of points############

# create one-dimensional arrays for x and y
x = np.arange(minx, maxx, res)
y = np.arange(miny, maxy, res)
# create the mesh based on these arrays
X, Y = np.meshgrid(x, y)
X = X.reshape((np.prod(X.shape),))
Y = Y.reshape((np.prod(Y.shape),))
coords = list(zip(X, Y))
dataframe=pd.DataFrame(columns=['X','Y','Elevation','Slope','Population','LandCover','RoadDistance'])
# improvements: add rivers and protected areas
dataframe['X']=X
dataframe['Y']=Y

#create geodataframe out of dataframe and clip it with the area (before it was a square)
geodataframe=gpd.GeoDataFrame(dataframe, geometry=gpd.points_from_xy(dataframe.X,dataframe.Y),crs=crs)
geodataframe=gpd.clip(geodataframe,area)

###########import vector files ##########
streets=gpd.read_file('Output/streets/edges.shp')
streets=streets.to_crs(crs)
streets_points=[]
for line in streets['geometry']:
    for x in list(zip(line.xy[0],line.xy[1])):
        streets_points.append(x)
streets_multipoint=MultiPoint(streets_points)
######## Manage raster files and sample their values ##############
# extract zipped files #
with zipfile.ZipFile('Output/Elevation.zip', 'r') as zip_ref:
    zip_ref.extractall('Output/Elevation')
with zipfile.ZipFile('Output/LandCover.zip', 'r') as zip_ref:
    zip_ref.extractall('Output/LandCover')
with zipfile.ZipFile('Output/Population.zip', 'r') as zip_ref:
    zip_ref.extractall('Output/Population')

#find the name of the raster files
file_elevation = os.listdir('Output/Elevation')
file_land_cover = os.listdir('Output/LandCover')
file_population = os.listdir('Output/Population')

#open files with rasterio
raster_elevation = rasterio.open('Output/Elevation/'+file_elevation[0])
raster_land_cover = rasterio.open('Output/LandCover/'+file_land_cover[0])
raster_population = rasterio.open('Output/Population/'+file_population[0])

#compute slope
gdal.DEMProcessing('Output/Elevation/slope.tif', 'Output/Elevation/'+file_elevation[0], 'slope')
raster_slope = rasterio.open('Output/Elevation/slope.tif')

#resample continuous raster data according to the desired resolution (bilinear method)
def resample(raster,res):
    #get pixel size
    affine = raster.transform
    pixelX = affine[0]
    pixelY =-affine[4]

    scale_factor = pixelX / res
    # resample data to target shape
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

    return data,profile

data_elevation,profile_elevation=resample(raster_elevation,res)
data_slope,profile_slope = resample(raster_slope,res)
#write resampled values in a new raster
with rasterio.open('Output/Elevation/elevation_resampled.tif', 'w', **profile_elevation) as dst:
#    dst.write(data_elevation.astype(rasterio.int16))
    dst.write(data_elevation)
with rasterio.open('Output/Elevation/slope_resampled.tif', 'w', **profile_slope) as dst:
#    dst.write(data_slope.astype(rasterio.float32))
    dst.write(data_slope)
#reopen resampled values. It seems stupid, need to check for a faster wat
raster_elevation_resampled = rasterio.open('Output/Elevation/elevation_resampled.tif')
raster_slope_resampled = rasterio.open('Output/Elevation/slope_resampled.tif')
#read arrays of rasters
elevation_resampled = raster_elevation_resampled.read()
slope_resampled = raster_slope_resampled.read()
land_cover=raster_land_cover.read()
population = raster_population.read()
#get population cell size to understand how to sample it
affine_pop = raster_population.transform
pixelX_pop = affine_pop[0]
pixelY_pop =-affine_pop[4]


#sample raster data with points of the regular grid
print('population assignment')
for i, row in geodataframe.iterrows():
    #population
    if pixelX_pop<=res:
        row_min,column_max = raster_population.index(row.X + res/2,row.Y + res/2)
        row_max, column_min = raster_population.index(row.X - res / 2, row.Y - res / 2)
        value = population[0,row_min:row_max,column_min:column_max].sum().sum()
        geodataframe.loc[i,'Population'] = value
    else:
        x, y = raster_population.index(row.X, row.Y)
        #no better assign to the center
        geodataframe.loc[i, 'Population'] = population[0, x, y]/pixelX_pop*res
    #road distance
    # dataframe.loc[i,'RoadDistance'] = np.min([Point(row.X,row.Y).distance(line) for line in streets.geometry]) too long!
    nearest_geoms = nearest_points(Point(row.X,row.Y), streets_multipoint)
    geodataframe.loc[i, 'RoadDistance'] = nearest_geoms[0].distance(nearest_geoms[1])

    #elevation
    x, y = raster_elevation_resampled.index(row.X, row.Y)
    geodataframe.loc[i, 'Elevation'] = elevation_resampled[0, x, y]

    #slope
    x, y = raster_slope_resampled.index(row.X, row.Y)
    geodataframe.loc[i, 'Slope'] = slope_resampled[0, x, y]

    #land cover
    x, y = raster_land_cover.index(row.X, row.Y)
    geodataframe.loc[i, 'LandCover'] = land_cover[0,x,y]

    print(i)


#dataframe.to_csv('Output/Namajavira_grid.csv')

#geodataframe.to_file('Output/Namajavira_grid1000.shp') #not able to save it (gives errors in the data types)
dataframe = pd.DataFrame(geodataframe.drop(columns='geometry'))
dataframe.to_csv('Output/Namajavira_grid_200.csv')

a=1