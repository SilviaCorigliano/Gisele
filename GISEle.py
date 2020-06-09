"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Running Code

In order to run the following algorithm please check if the Input file is
configured accordingly, for more information check the README.
"""

import supporting_GISEle2
from Codes import create_df, clustering

"Introduction"
supporting_GISEle2.l()
print('Welcome to GISEle! From which step you want to start?')
supporting_GISEle2.l()
steps = ['1.Assigning weights, Cleaning and creating the GeoDataFrame',
         '2.Clustering',
         '3.Grid creation',
         '4.Load creation',
         '5.Generation sizing',
         '6.Final results']
print("\n".join(steps))
supporting_GISEle2.s()
# step = int(input('Which step do you want to select?: '))
step = 1
if step == 1:
    '-------------------------------------------------------------------------'
    "1. Assigning weights, Cleaning and Creating the GeoDataFrame"

    df, input_sub, input_csv, crs, resolution, unit, pop_load, pop_thresh, \
        line_bc, limit_hv, limit_mv = create_df.import_csv_file(step)

    df_weighted = create_df.weighting(df)

    geo_df, pop_points = create_df.creating_geodataframe(df_weighted, crs,
                                                         unit, input_csv, step)
    '-------------------------------------------------------------------------'
    "2.Clustering"

    clustering.cluster_sensitivity(pop_points, geo_df)

    geo_df_clustered, clusters_list = clustering.cluster_analysis(pop_points,
                                                                  geo_df,
                                                                  pop_load)
    '-------------------------------------------------------------------------'
    "3.Grid creation"

    grid_resume = supporting_GISEle2.grid(geo_df_clustered, geo_df,
                                          crs, clusters_list,
                                          resolution, pop_thresh,
                                          input_sub, line_bc,
                                          limit_hv, limit_mv)

    grid_optimized = supporting_GISEle2.grid_optimization(geo_df_clustered,
                                                          geo_df, grid_resume,
                                                          crs, resolution,
                                                          line_bc)
    '-------------------------------------------------------------------------'

    "4.Load creation"

    loads_list, houses_list, house_ref = supporting_GISEle2.load()
    '-------------------------------------------------------------------------'
    "5.Generation sizing"

    total_energy, mg_NPC = supporting_GISEle2.sizing(loads_list, clusters_list,
                                                     houses_list, house_ref)

    '-------------------------------------------------------------------------'
    "6.Final results"
    final_LCOEs = supporting_GISEle2.final_results(clusters_list, total_energy,
                                                   grid_resume, mg_NPC)

elif step == 2:
    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv = \
        supporting_GISEle2.import_csv_file(step)

    geo_df, pop_points = supporting_GISEle2. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "2.Clustering"

    supporting_GISEle2.clustering_sensitivity(pop_points, geo_df)
    geo_df_clustered, clusters_list = \
        supporting_GISEle2.clustering(pop_points, geo_df, pop_load)
    '-------------------------------------------------------------------------'
    "3.Grid creation"
    grid_resume = supporting_GISEle2.grid(geo_df_clustered, geo_df,
                                          crs, clusters_list,
                                          resolution, pop_thresh,
                                          input_sub, line_bc,
                                          limit_hv, limit_mv)

    grid_optimized = supporting_GISEle2.grid_optimization(geo_df_clustered,
                                                          geo_df, grid_resume,
                                                          crs, resolution,
                                                          line_bc)

elif step == 3:
    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv, geo_df_clustered, \
        clusters_list, = supporting_GISEle2.import_csv_file(step)

    geo_df, pop_points = supporting_GISEle2. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "3. Grid creation"

    grid_resume = supporting_GISEle2.grid(geo_df_clustered, geo_df,
                                          crs, clusters_list,
                                          resolution, pop_thresh,
                                          input_sub, line_bc,
                                          limit_hv, limit_mv)

    grid_optimized = supporting_GISEle2.grid_optimization(geo_df_clustered,
                                                          geo_df, grid_resume,
                                                          crs, resolution,
                                                          line_bc)

elif step == 4:
    '-------------------------------------------------------------------------'
    "4.Load creation"

    loads_list, houses_list, house_ref = supporting_GISEle2.load()
    '-------------------------------------------------------------------------'
    "5.Generation sizing"

    clusters_list = []
    total_energy, mg_NPC = supporting_GISEle2.sizing(loads_list, clusters_list,
                                                     houses_list, house_ref)

    '-------------------------------------------------------------------------'
    "6.Final results"
    grid_resume = []

    final_LCOEs = supporting_GISEle2.final_results(clusters_list, total_energy,
                                                   grid_resume, mg_NPC)
