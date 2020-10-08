import ee
import os
import shutil
import osmnx as ox
import geopandas as gpd
import requests
import pandas as pd
import supporting_GISEle2
from shapely.geometry import Point, LineString
import numpy as np


def data_gathering(crs, study_area):

    #  Import layer with region boundaries and extract its extent
    #  the study area will be given by the dash app later, for now its manually

    # study_area = gpd.read_file(r'Input/bolivia.shp')
    # study_area = gpd.read_file(r'Input/Namanjavira_4326.geojson')

    study_area = study_area.to_crs(4326)
    min_x, min_y, max_x, max_y = study_area.geometry.total_bounds
    if os.path.exists("Output\Datasets\Elevation"):
        shutil.rmtree(r'Output\Datasets\Elevation')
    os.mkdir(r'Output\Datasets\Elevation')
    if os.path.exists("Output\Datasets\Population"):
        shutil.rmtree(r'Output\Datasets\Population')
    os.mkdir(r'Output\Datasets\Population')
    if os.path.exists("Output\Datasets\Landcover"):
        shutil.rmtree(r'Output\Datasets\LandCover')
    os.mkdir(r'Output\Datasets\LandCover')
    # Download layers from Earth Engine

    ee.Initialize()

    # DEM with 30m resolution SRTM
    print('Elevation Database:')
    image = ee.Image("USGS/SRTMGL1_003")
    out_path = str('Output\Datasets\Elevation\Elevation.zip')
    scale = 30
    supporting_GISEle2.download_tif(study_area, crs, scale, image, out_path)

    """
    Land Cover from Copernicus
    Copernicus Global Land Cover Layers: CGLS-LC100 collection 2.
    100m resolution. Ref year 2015
    download only images related to discrete classification(see documentation)
    """

    print('Land Cover Database:')
    collection = ee.ImageCollection(
        "COPERNICUS/Landcover/100m/Proba-V/Global").select(
        'discrete_classification')
    image = collection.sort('system:time_start', False) \
        .first()
    out_path = 'Output\Datasets\LandCover\LandCover.zip'
    scale = 100
    supporting_GISEle2.download_tif(study_area, crs, scale, image, out_path)

    # Population from worldpop
    print('Population Database:')
    scale = 100
    image = ee.ImageCollection("WorldPop/GP/100m/pop").filter(
        ee.Filter.date('2015-01-01', '2015-12-31')).select('population')
    image = image.reduce(ee.Reducer.median())
    out_path = 'Output\Datasets\Population\Population.zip'
    supporting_GISEle2.download_tif(study_area, crs, scale, image, out_path)

    # Population from JRC
    # scale=250
    # image = ee.ImageCollection("JRC/GHSL/P2016/POP_GPW_GLOBE_V1").\
    #     sort('system:time_start', False).first().select('population_count')
    # protected areas from WDPA (as polygons)
    # protect_areas = ee.FeatureCollection("WCMC/WDPA/current/polygons").\
    #     getDownloadURL('csv')
    # how to save it as shp?

    # Download road layers from OpenStreetMaps
    print('Downloading roads, transmission grid '
          'and substations from OpenStreetMap..')
    graph = ox.graph_from_bbox(min_y, max_y,
                               min_x, max_x, network_type='drive_service')
    ox.save_graph_shapefile(graph,
                            filepath='Output/Datasets/Roads')  # crs is 4326

    # # Transmission grid and substations
    #
    # overpass_url = "http://overpass-api.de/api/interpreter"
    # overpass_query = """
    # [out:json];
    # (
    #  way["power"="line"](""" + str(min_y) + ',' + str(min_x) + ',' + str(
    #     max_y) + ',' + str(max_x) + ''');
    # );
    # out geom;
    # '''
    # response = requests.get(overpass_url,
    #                         params={'data': overpass_query})
    # data = response.json()
    # df = pd.json_normalize(data['elements'])
    # if not df.empty:
    #     df['geometry_new'] = df['geometry']
    #     for i, row in df.iterrows():
    #         points = []
    #         for j in row['geometry']:
    #             points.append(Point(j['lon'], j['lat']))
    #         df.loc[i, 'geometry_new'] = LineString(points)
    #     geo_df = gpd.GeoDataFrame(
    #         df[['id', 'tags.power', 'tags.voltage', 'geometry_new']],
    #         geometry='geometry_new', crs='epsg:4326')
    #     geo_df.to_crs(crs)
    #     geo_df.to_file('Output/Datasets/transmission_grid.shp')
    #
    # else:
    #     print('ERROR: Transmission grid data could not be downloaded')
    #
    # overpass_query = """
    # [out:json];
    # (node["power"="substation"](""" + str(min_y) + ',' + str(
    #     min_x) + ',' + str(max_y) + ',' + str(max_x) + ''');
    #  way["power"="substation"](''' + str(min_y) + ',' + str(min_x) + ',' + str(
    #     max_y) + ',' + str(max_x) + ''');
    # );
    # out center;
    # '''
    # response = requests.get(overpass_url, params={'data': overpass_query})
    # data = response.json()
    # df = pd.json_normalize(data['elements'])
    # if not df.empty:
    #     # Collect coordinates into list
    #     coordinates = []
    #     for element in data['elements']:
    #         if element['type'] == 'node':
    #             lon = element['lon']
    #             lat = element['lat']
    #             coordinates.append((lon, lat))
    #         elif 'center' in element:
    #             lon = element['center']['lon']
    #             lat = element['center']['lat']
    #             coordinates.append((lon, lat))
    #     # Convert coordinates into numpy array
    #     coordinates = np.array(coordinates)
    #     geo_df = gpd.GeoDataFrame(
    #         df.drop('nodes', axis=1), crs='epsg:4326',
    #         geometry=gpd.points_from_xy(coordinates[:, 0], coordinates[:, 1]))
    #     geo_df.to_crs(crs)
    #     geo_df.to_file('Output/Datasets/substations.shp')
    # else:
    #     print('ERROR: Substation data could not be downloaded')
    return
