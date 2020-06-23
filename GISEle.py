"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Running Code

In order to run the following algorithm please check if the Input file is
configured accordingly, for more information check the README.
"""

import supporting_GISEle2
from Codes import initialization, clustering, grid, branches, optimization

"Introduction"
supporting_GISEle2.l()
print('Welcome to GISEle! From which step you want to start?')
supporting_GISEle2.l()
steps = ['1.Assigning weights, Cleaning and creating the GeoDataFrame',
         '2.Clustering',
         '3.Grid creation',
         "4.Main branch and collateral's",
         '5.Load creation',
         '6.Generation sizing',
         '7.Final results']
print("\n".join(steps))
supporting_GISEle2.s()
# step = int(input('Which step do you want to select?: '))
step = 4
if step == 1:
    '-------------------------------------------------------------------------'
    "1. Assigning weights, Cleaning and Creating the GeoDataFrame"

    df, input_sub, input_csv, crs, resolution, unit, pop_load, pop_thresh, \
        line_bc, limit_hv, limit_mv = initialization.import_csv_file(step)

    df_weighted = initialization.weighting(df)

    geo_df, pop_points = initialization.creating_geodataframe(df_weighted, crs,
                                                              unit, input_csv,
                                                              step)
    '-------------------------------------------------------------------------'
    "2.Clustering"

    clustering.sensitivity(pop_points, geo_df)

    geo_df_clustered, clusters_list = clustering.analysis(pop_points, geo_df,
                                                          pop_load)
    '-------------------------------------------------------------------------'
    "3.Grid creation"

    grid_resume = grid.routing(geo_df_clustered, geo_df, clusters_list,
                               resolution, pop_thresh, input_sub, line_bc,
                               limit_hv, limit_mv)

    grid_optimized = optimization.connections(geo_df, grid_resume, resolution,
                                              line_bc, limit_hv, limit_mv,
                                              step)
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
        initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "2.Clustering"

    clustering.sensitivity(pop_points, geo_df)
    geo_df_clustered, clusters_list = \
        clustering.analysis(pop_points, geo_df, pop_load)
    '-------------------------------------------------------------------------'
    "3.Grid creation"
    grid_resume = grid.routing(geo_df_clustered, geo_df, clusters_list,
                               resolution, pop_thresh, input_sub, line_bc,
                               limit_hv, limit_mv)

    grid_optimized = optimization.connections(geo_df, grid_resume, resolution,
                                              line_bc, limit_hv, limit_mv,
                                              step)

elif step == 3:
    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv, geo_df_clustered, \
        clusters_list, = initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "3. Grid creation"

    grid_resume = grid.routing(geo_df_clustered, geo_df, clusters_list,
                               resolution, pop_thresh, input_sub, line_bc,
                               limit_hv, limit_mv)

    grid_optimized = optimization.connections(geo_df, grid_resume, resolution,
                                              line_bc, limit_hv, limit_mv,
                                              step)


elif step == 4:

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv, geo_df_clustered, \
        clusters_list, input_csv_lr, pop_thresh_lr, line_bc_col, full_ele = \
        initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    grid_resume = branches.routing(geo_df_clustered, geo_df, clusters_list,
                                   resolution, pop_thresh, input_sub, line_bc,
                                   limit_hv, limit_mv, pop_load, input_csv_lr,
                                   pop_thresh_lr, line_bc_col, full_ele)

    grid_optimized = optimization.connections(geo_df, grid_resume, resolution,
                                              line_bc, limit_hv, limit_mv,
                                              step)
elif step == 5:
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
