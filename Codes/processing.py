import zipfile
import os
import rasterio.mask
from osgeo import gdal
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points


def create_mesh(study_area, crs, resolution):
    # crs = 32737
    print('Processing the imported data and creating the input csv file..')
    df = pd.DataFrame(columns=['ID', 'X', 'Y', 'Elevation', 'Slope',
                               'Population', 'Land_cover', 'Road_dist',
                               'River_flow'])

    study_area = study_area.to_crs(crs)
    min_x, min_y, max_x, max_y = study_area.geometry.total_bounds
    # re-project layer according to metric coordinate system
    # create one-dimensional arrays for x and y
    lon = np.arange(min_x, max_x, resolution)
    lat = np.arange(min_y, max_y, resolution)
    lon, lat = np.meshgrid(lon, lat)

    # improvements: add rivers and protected areas
    df['X'] = lon.reshape((np.prod(lon.shape),))
    df['Y'] = lat.reshape((np.prod(lat.shape),))

    # create geo-data-frame out of df and clip it with the area
    geo_df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.X, df.Y),
                              crs=crs)
    geo_df = gpd.clip(geo_df, study_area)
    geo_df.reset_index(drop=True, inplace=True)
    geo_df['ID'] = geo_df.index
    # import vector files
    streets = gpd.read_file('Output/Datasets/Roads/edges.shp')
    streets = streets.to_crs(crs)
    streets_points = []
    for line in streets['geometry']:
        for x in list(zip(line.xy[0], line.xy[1])):
            streets_points.append(x)
    streets_multipoint = MultiPoint(streets_points)
    # Manage raster files and sample their values
    # extract zipped files #
    with zipfile.ZipFile('Output/Datasets/Elevation.zip', 'r') as zip_ref:
        zip_ref.extractall('Output/Datasets/Elevation')
    with zipfile.ZipFile('Output/Datasets/LandCover.zip', 'r') as zip_ref:
        zip_ref.extractall('Output/Datasets/LandCover')
    with zipfile.ZipFile('Output/Datasets/Population.zip', 'r') as zip_ref:
        zip_ref.extractall('Output/Datasets/Population')

    # find the name of the raster files
    file_elevation = os.listdir('Output/Datasets/Elevation')
    file_land_cover = os.listdir('Output/Datasets/LandCover')
    file_population = os.listdir('Output/Datasets/Population')

    # open files with rasterio
    raster_elevation = rasterio.open('Output/Datasets/Elevation/'
                                     + file_elevation[0])
    raster_land_cover = rasterio.open('Output/Datasets/LandCover/'
                                      + file_land_cover[0])
    raster_population = rasterio.open('Output/Datasets/Population/'
                                      + file_population[0])

    # compute slope
    gdal.DEMProcessing('Output/Datasets/Elevation/slope.tif',
                       'Output/Datasets/Elevation/' + file_elevation[0],
                       'slope')
    raster_slope = rasterio.open('Output/Datasets/Elevation/slope.tif')

    # re-sample continuous raster according to the desired resolution(bilinear)

    data_elevation, profile_elevation = resample(raster_elevation, resolution)
    data_slope, profile_slope = resample(raster_slope, resolution)

    # write re-sampled values in a new raster
    with rasterio.open('Output/Datasets/Elevation/elevation_resampled.tif',
                       'w', **profile_elevation) as dst:
        dst.write(data_elevation)
        # dst.write(data_elevation.astype(rasterio.int16))
    with rasterio.open('Output/Datasets/Elevation/slope_resampled.tif', 'w',
                       **profile_slope) as dst:
        dst.write(data_slope)
        # dst.write(data_slope.astype(rasterio.float32))
    # reopen resampled values. It seems stupid, need to check for a faster way
    raster_elevation_resampled = \
        rasterio.open('Output/Datasets/Elevation/elevation_resampled.tif')
    raster_slope_resampled = \
        rasterio.open('Output/Datasets/Elevation/slope_resampled.tif')

    # read arrays of raster layers
    elevation_resampled = raster_elevation_resampled.read()
    slope_resampled = raster_slope_resampled.read()
    land_cover = raster_land_cover.read()
    population = raster_population.read()
    # get population cell size to understand how to sample it
    affine_pop = raster_population.transform
    pixel_x_pop = affine_pop[0]

    # sample raster data with points of the regular grid
    print('Assigning values to each point..')
    for i, row in geo_df.iterrows():
        # population
        if pixel_x_pop <= resolution:
            row_min, column_max = raster_population.\
                index(row.X + resolution/2, row.Y + resolution/2)
            row_max, column_min = raster_population.\
                index(row.X - resolution / 2, row.Y - resolution / 2)
            value = population[0, row_min:row_max,
                               column_min:column_max].sum().sum()
            geo_df.loc[i, 'Population'] = value
        else:
            x, y = raster_population.index(row.X, row.Y)
            # better assign to the center
            geo_df.loc[i, 'Population'] = \
                population[0, x, y]/pixel_x_pop * resolution
        # road distance
        nearest_geoms = nearest_points(Point(row.X, row.Y), streets_multipoint)
        geo_df.loc[i, 'Road_dist'] = \
            nearest_geoms[0].distance(nearest_geoms[1])

        # elevation
        x, y = raster_elevation_resampled.index(row.X, row.Y)
        geo_df.loc[i, 'Elevation'] = elevation_resampled[0, x, y]

        # slope
        x, y = raster_slope_resampled.index(row.X, row.Y)
        geo_df.loc[i, 'Slope'] = slope_resampled[0, x, y]

        # land cover
        x, y = raster_land_cover.index(row.X, row.Y)
        geo_df.loc[i, 'Land_cover'] = land_cover[0, x, y]

        print('\r' + str(i) + '/' + str(geo_df.index.__len__()),
              sep=' ', end='', flush=True)

    print('\n')
    geo_df.Elevation = geo_df.Elevation.astype(int)
    pd.DataFrame(geo_df.drop(columns='geometry'))\
        .to_csv('Output/imported_csv.csv', index=False)

    return geo_df


def resample(raster, resolution):

    # get pixel size
    affine = raster.transform
    pixel_x = affine[0]
    scale_factor = pixel_x / resolution
    # re-sample data to target shape
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
