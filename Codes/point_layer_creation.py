# Run the following code in QGIS Python Console in order to create the point layer

import sys
import os
import qgis
from qgis.core import *
from PyQt5.QtGui import *
from processing.core.Processing import Processing
Processing.initialize()
import processing

###########################################################################
# Things that need to be changed
# This is the workspace with all the datasets that you wish to use, they should be here in corresponding sub-folders.
# The workspace has to include the subfolders with all of the necessary layers
workspace = r'C:\Users\User\Desktop\thesis\Pesi\Datasets'

# Name of subfolders contained in workspace must be: Admin, Population, Land_cover, DEM, Roads, Water_bodies, Rivers, Natural_reserves and Transmission_network

# The name of the settlement that you are analysing: this will be name of your output csv file
settlements_fc = 'Namajavira_west_electric_borders'

# Please, pay attention to provide all input layers with the same crs

# Which is the desired resolution of your point layer? (distance between points)
# NB: Pay attention to resolution value!!! It should be coherent with the crs of your layers
res = 200
# repeat resolution as string
resol = '200'

# Which is the epsg code of the coordinate system that you wish to project your datasets TO?
projCoord = "EPSG:32737"

########################################################################################

# The naming of all the datasets. make sure that the datasets are named as they are named here!
# All datasets should be provided with the same crs

# Population, elevation, land cover and river flow rate should be raster files
# elevation expressed in meters, rivers' flow rate in m3/s
pop = 'Population'
elevation = 'Elevation'
land_cover = 'Land_cover'
river_flow = 'River_flow'
planned = 'planned_grid'
exisiting = 'existing_grid'
# Admin and roads should be shapefiles
admin = settlements_fc
roads = 'Roads'
water_bodies = 'Water_bodies'

# Import admin polygon
# We import it in order to clip the population dataset (in case the population dataset is global)
admin = workspace + r'/Admin/' + admin + '.shp'
processing.run("native:reprojectlayer", {'INPUT':admin,'TARGET_CRS':projCoord,'OUTPUT':workspace +r'/Admin/' + settlements_fc + '_proj.shp'})
adminproj = workspace +r'/Admin/' + settlements_fc + '_proj.shp'

# Clip population map with admin, and resample it in the desired resolution
pop_data = workspace + r"/Population/"+ pop + ".tif"
processing.run("gdal:cliprasterbymasklayer", {'INPUT':pop_data,'MASK':admin,'SOURCE_CRS':None,'TARGET_CRS':None,'NODATA':None,'ALPHA_BAND':False,'CROP_TO_CUTLINE':True,'KEEP_RESOLUTION':False,'SET_RESOLUTION':False,'X_RESOLUTION':None,'Y_RESOLUTION':None,'MULTITHREADING':False,'OPTIONS':'','DATA_TYPE':0,'OUTPUT':workspace + r'/Population/' + pop + settlements_fc[0:3] +'.tif'})
processing.run("gdal:warpreproject", {'INPUT':workspace + r'/Population/' + pop + settlements_fc[0:3] +'.tif','SOURCE_CRS':None,'TARGET_CRS':projCoord,'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':'','DATA_TYPE':0,'TARGET_EXTENT':None,'TARGET_EXTENT_CRS':None,'MULTITHREADING':False,'EXTRA':'','OUTPUT':workspace + r'/Population/' + pop + settlements_fc[0:3] +'_proj.tif'})
processing.run("grass7:r.resamp.stats", {'input':workspace + r'/Population/' + pop + settlements_fc[0:3] +'_proj.tif','method':8,'quantile':0.5,'-n':False,'-w':True,'output':workspace + r'/Population/' + pop + settlements_fc[0:3] +'_projandres.tif','GRASS_REGION_PARAMETER':None,'GRASS_REGION_CELLSIZE_PARAMETER':res,'GRASS_RASTER_FORMAT_OPT':'','GRASS_RASTER_FORMAT_META':''})

# Create point layer based on Population
if not os.path.exists(workspace + r"\Points_layer"):
     os.makedirs(workspace + r"\Points_layer")
Pointspath = workspace + r'\Points_layer'
processing.run("saga:rastervaluestopoints", {'GRIDS':[workspace + r'/Population/' + pop + settlements_fc[0:3] +'_projandres.tif'],'POLYGONS':adminproj,'NODATA':False,'TYPE':0,'SHAPES':Pointspath + r'\Points_1_' + settlements_fc[0:3] + '.shp'})

elevation_data = workspace + r'/DEM/' + elevation + '.tif'
# Processing of DEM layer, with filling of NoData
processing.run("gdal:cliprasterbymasklayer", {'INPUT':elevation_data,'MASK':admin,'SOURCE_CRS':None,'TARGET_CRS':None,'NODATA':None,'ALPHA_BAND':False,'CROP_TO_CUTLINE':True,'KEEP_RESOLUTION':False,'SET_RESOLUTION':False,'X_RESOLUTION':None,'Y_RESOLUTION':None,'MULTITHREADING':False,'OPTIONS':'','DATA_TYPE':0,'OUTPUT':workspace + r'/DEM/' + elevation + settlements_fc[0:3] +'.tif'})
processing.run("gdal:warpreproject", {'INPUT':workspace + r'/DEM/' + elevation + settlements_fc[0:3] +'.tif','SOURCE_CRS':None,'TARGET_CRS':projCoord,'RESAMPLING':1,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':'','DATA_TYPE':0,'TARGET_EXTENT':None,'TARGET_EXTENT_CRS':None,'MULTITHREADING':False,'EXTRA':'','OUTPUT':workspace + r'/DEM/' + elevation + settlements_fc[0:3] +'_proj.tif'})
processing.run("gdal:fillnodata", {'INPUT':workspace + r'/DEM/' + elevation + settlements_fc[0:3] +'_proj.tif','BAND':1,'DISTANCE':7,'ITERATIONS':0,'NO_MASK':False,'MASK_LAYER':None,'OUTPUT':workspace + r'/DEM/' + elevation + settlements_fc[0:3] +'_filled.tif'})

# Creation of Slope layer
if not os.path.exists(workspace + r"/Slope"):
     os.makedirs(workspace + r"/Slope")
processing.run("gdal:slope", {'INPUT':workspace + r'/DEM/' + elevation + settlements_fc[0:3] +'_filled.tif','BAND':1,'SCALE':1,'AS_PERCENT':False,'COMPUTE_EDGES':False,'ZEVENBERGEN':False,'OPTIONS':'','OUTPUT':workspace + r'/Slope/Slope_' + settlements_fc[0:3] +'.tif'})

# Processing of Land_cover layer
processing.run("gdal:cliprasterbymasklayer", {'INPUT':workspace + r'/Land_cover/' + land_cover + '.tif','MASK':admin,'SOURCE_CRS':None,'TARGET_CRS':None,'NODATA':None,'ALPHA_BAND':False,'CROP_TO_CUTLINE':True,'KEEP_RESOLUTION':False,'SET_RESOLUTION':False,'X_RESOLUTION':None,'Y_RESOLUTION':None,'MULTITHREADING':False,'OPTIONS':'','DATA_TYPE':0,'OUTPUT':workspace + r'/Land_cover/' + land_cover + settlements_fc[0:3] +'.tif'})
processing.run("gdal:warpreproject", {'INPUT':workspace + r'/Land_cover/' + land_cover + settlements_fc[0:3] +'.tif','SOURCE_CRS':None,'TARGET_CRS':projCoord,'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':'','DATA_TYPE':0,'TARGET_EXTENT':None,'TARGET_EXTENT_CRS':None,'MULTITHREADING':False,'EXTRA':'','OUTPUT':workspace + r'/Land_cover/' + land_cover + settlements_fc[0:3] +'_proj.tif'})

# Processing of rivers flow rate
processing.run("gdal:cliprasterbymasklayer", {'INPUT':workspace + r'/Rivers/' + river_flow + '.tif','MASK':admin,'SOURCE_CRS':None,'TARGET_CRS':None,'NODATA':None,'ALPHA_BAND':False,'CROP_TO_CUTLINE':True,'KEEP_RESOLUTION':False,'SET_RESOLUTION':False,'X_RESOLUTION':None,'Y_RESOLUTION':None,'MULTITHREADING':False,'OPTIONS':'','DATA_TYPE':0,'OUTPUT':workspace + r'/Rivers/' + river_flow + settlements_fc[0:3] +'.tif'})
processing.run("gdal:warpreproject", {'INPUT':workspace + r'/Rivers/' + river_flow + settlements_fc[0:3] +'.tif','SOURCE_CRS':None,'TARGET_CRS':projCoord,'RESAMPLING':0,'NODATA':None,'TARGET_RESOLUTION':None,'OPTIONS':'','DATA_TYPE':0,'TARGET_EXTENT':None,'TARGET_EXTENT_CRS':None,'MULTITHREADING':False,'EXTRA':'','OUTPUT':workspace + r'/Rivers/' + river_flow + settlements_fc[0:3] +'_proj.tif'})
processing.run("grass7:r.resamp.stats", {'input':workspace + r'/Rivers/' + river_flow + settlements_fc[0:3] +'_proj.tif','method':0,'quantile':0.5,'-n':False,'-w':True,'output':workspace + r'/Rivers/' + river_flow + settlements_fc[0:3] +'_projandres.tif','GRASS_REGION_PARAMETER':None,'GRASS_REGION_CELLSIZE_PARAMETER':res,'GRASS_RASTER_FORMAT_OPT':'','GRASS_RASTER_FORMAT_META':''})

# Let's now assign all of this layer's values to the point layer
# NB: Elevation and Slope, which are continuous layers, will be resampled with bilinear method (1), whereas Land_cover (discrete layer) will be resampled with nearest neighbor method
Points = QgsVectorLayer(Pointspath + r'/Points_1_' + settlements_fc[0:3] + '.shp','','ogr')
elevation = QgsRasterLayer(workspace + r'/DEM/' + elevation + settlements_fc[0:3] +'_filled.tif','Elevation')
slope = QgsRasterLayer(workspace + r'/Slope/Slope_' + settlements_fc[0:3] +'.tif','Slope')
landcover = QgsRasterLayer(workspace + r'/Land_cover/' + land_cover + settlements_fc[0:3] +'_proj.tif','Land_cover')
riverflow = QgsRasterLayer(workspace + r'/Rivers/' + river_flow + settlements_fc[0:3] +'_projandres.tif','River_flow')

processing.run("saga:addrastervaluestopoints", {'SHAPES':Points,'GRIDS':[elevation, slope],'RESAMPLING':1,'RESULT':Pointspath + r'\Points_2_' + settlements_fc[0:3] + '.shp'})
processing.run("saga:addrastervaluestopoints", {'SHAPES':Pointspath + r'\Points_2_' + settlements_fc[0:3] + '.shp','GRIDS':[landcover, riverflow],'RESAMPLING':0,'RESULT':Pointspath + r'\Points_3_' + settlements_fc[0:3] + '.shp'})

# Water bodies processing
processing.run("native:clip", {'INPUT':workspace + r'/Water_bodies/' + water_bodies + '.shp','OVERLAY':admin,'OUTPUT':workspace + r'/Water_bodies/' + water_bodies + settlements_fc[0:3] + '.shp'})
processing.run("native:reprojectlayer", {'INPUT':workspace + r'/Water_bodies/' + water_bodies + settlements_fc[0:3] + '.shp','TARGET_CRS':projCoord,'OUTPUT':workspace + r'/Water_bodies/' + water_bodies + settlements_fc[0:3] +'_proj.shp'})
processing.run("qgis:joinattributesbylocation", {'INPUT':Pointspath + r'/Points_3_' + settlements_fc[0:3] + '.shp','JOIN':workspace + r'/Water_bodies/' + water_bodies + settlements_fc[0:3] +'_proj.shp','PREDICATE':[5],'JOIN_FIELDS':['NAME'],'METHOD':0,'DISCARD_NONMATCHING':False,'PREFIX':'','OUTPUT':Pointspath + r'\Points_4_' + settlements_fc[0:3] + '.shp'})


# In the end, we have to determine distance from each point to the nearest road and assign this value to the point layer
processing.run("native:clip", {'INPUT':workspace + r'/Roads/' + roads + '.shp','OVERLAY':admin,'OUTPUT':workspace + r'/Roads/' + roads + settlements_fc[0:3] +'.shp'})
processing.run("native:reprojectlayer", {'INPUT':workspace + r'/Roads/' + roads + settlements_fc[0:3] +'.shp','TARGET_CRS':projCoord,'OUTPUT':workspace + r'/Roads/' + roads + settlements_fc[0:3] +'_proj.shp'})
processing.run("qgis:addfieldtoattributestable", {'INPUT':Pointspath + r'/Points_4_' + settlements_fc[0:3] + '.shp','FIELD_NAME':'Road_dist','FIELD_TYPE':1,'FIELD_LENGTH':18,'FIELD_PRECISION':10,'OUTPUT':Pointspath + r'/Points_5a_' + settlements_fc[0:3] + '.shp'})
processing.run("grass7:v.distance", {'from':Pointspath + r'/Points_5a_' + settlements_fc[0:3] + '.shp','from_type':[0,1,3],'to':workspace + r'/Roads/' + roads + settlements_fc[0:3] +'_proj.shp','to_type':[0,1,3],'dmax':-1,'dmin':-1,'upload':[1],'column':['Road_dist'],'to_column':None,'from_output':Pointspath + r'/Points_5b_' + settlements_fc[0:3] + '.shp','output':workspace + r'/Roads/distancefrompoint' + '.shp','GRASS_REGION_PARAMETER':None,'GRASS_SNAP_TOLERANCE_PARAMETER':-1,'GRASS_MIN_AREA_PARAMETER':0.0001,'GRASS_OUTPUT_TYPE_PARAMETER':0,'GRASS_VECTOR_DSCO':'','GRASS_VECTOR_LCO':'','GRASS_VECTOR_EXPORT_NOCAT':False})
processing.run("qgis:deletecolumn", {'INPUT':Pointspath + r'/Points_5b_' + settlements_fc[0:3] + '.shp','COLUMN':['cat'],'OUTPUT':Pointspath + r'/Points_5c_' + settlements_fc[0:3] + '.shp'})

# Assign coordinates to points
settlements = QgsVectorLayer(Pointspath + r'/Points_5c_' + settlements_fc[0:3] + '.shp',"","ogr")

# Save as a csv file
workspace2 = r'C:\Users\User\Desktop\thesis\Pesi'
settlements.setName(settlements_fc)
if not os.path.exists(workspace2 + r"\Results"):
     os.makedirs(workspace2 + r"\Results")
QgsVectorFileWriter.writeAsVectorFormat(settlements, workspace2 + r"/Results/Points_final_" + settlements_fc + '_' + resol + ".csv", "utf-8", settlements.crs(), "CSV")