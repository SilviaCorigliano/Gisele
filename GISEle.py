"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Running Code

In order to run the following algorithm please check if the Input file is
configured accordingly, for more information check the README.
"""

import supporting_GISEle2

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
step=2
if step == 1:
    "1. Assigning weights, Cleaning and Creating the GeoDataFrame"

    df, input_sub, input_csv, crs, resolution, unit, pop_load, pop_thresh, \
        line_bc, limit_HV, limit_MV = supporting_GISEle2.import_csv_file(step)

    df_weighted = supporting_GISEle2.weighting(df)

    geo_df, pop_points = supporting_GISEle2.\
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)
    '-------------------------------------------------------------------------'
    "2.Clustering"

    supporting_GISEle2.clustering_sensitivity(pop_points, geo_df)

    gdf_clusters, clusters_list_2, clusters_load = supporting_GISEle2.\
        clustering(pop_points, geo_df,)

    '-------------------------------------------------------------------------'
    "7.Grid creation"

    grid_resume, paycheck = supporting_GISEle2.grid(gdf_clusters, geo_df, crs,
                                                    clusters_list_2,
                                                    resolution, clusters_load,
                                                    pop_thresh, input_sub,
                                                    line_bc, limit_HV,
                                                    limit_MV)
    '-------------------------------------------------------------------------'
    "8.Grid optimization"

    grid_optimized = supporting_GISEle2.grid_optimization(gdf_clusters, geo_df, grid_resume, crs,
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

elif step == 2:

    "1. Cleaning and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_HV, limit_MV = \
        supporting_GISEle2.import_csv_file(step)

    geo_df, pop_points = supporting_GISEle2. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-----------------------------------------------------------------------------------------------------------------'
    "7.Clustering and Grid creation"

    supporting_GISEle2.clustering_sensitivity(pop_points, geo_df)
    gdf_clusters, clusters_list_2, clusters_load = \
        supporting_GISEle2.clustering(pop_points, geo_df)

    grid_resume, paycheck = supporting_GISEle2.grid(gdf_clusters, geo_df, crs,
                                                    clusters_list_2,
                                                    resolution, clusters_load,
                                                    pop_thresh, input_sub,
                                                    line_bc, limit_HV,
                                                    limit_MV)

    "8.Grid optimization"

    grid_optimized = supporting_GISEle2.grid_optimization(gdf_clusters, geo_df, grid_resume, crs,
                                                          resolution, paycheck)

elif step == 3:

    "*.DataFrame preparation"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_HV, limit_MV,  gdf_clusters, \
        clusters_list_2, clusters_load = \
        supporting_GISEle2.import_csv_file(step)

    geo_df, total_points, total_people, pop_points = supporting_GISEle2. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-----------------------------------------------------------------------------------------------------------------'
    "7. Grid creation"

    # gdf_clusters, clusters_list_2, clusters_load = supporting_GISEle2.import_cluster()

    grid_resume, paycheck = supporting_GISEle2.grid(gdf_clusters, geo_df, crs,
                                                    clusters_list_2,
                                                    resolution, clusters_load,
                                                    pop_thresh, input_sub,
                                                    line_bc, limit_HV, limit_MV)

    "8.Grid optimization"

    grid_optimized = supporting_GISEle2.grid_optimization(gdf_clusters, geo_df, grid_resume, crs,
                                                          resolution, paycheck)

elif step == 4:

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
