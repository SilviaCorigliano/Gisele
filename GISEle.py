"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Running Code

In order to run the following algorithm please check if the Input files are
configured accordingly, for more information check the User Manual.
"""

import supporting_GISEle2
from Codes import initialization, clustering, grid, branches, optimization, \
    results

"Introduction"
supporting_GISEle2.l()
print('Welcome to GISEle! From which step you want to start?')
supporting_GISEle2.l()
steps = ['1.Assigning weights, Cleaning and creating the GeoDataFrame',
         '2.Clustering',
         '3.Grid creation',
         '4.Microgrid Sizing',
         '5.LCOE Analysis']
print("\n".join(steps))
supporting_GISEle2.s()
# step = int(input('Which step do you want to select?: '))
step = 3
if step == 1:
    '-------------------------------------------------------------------------'
    "1. Assigning weights, Cleaning and Creating the GeoDataFrame"

    df, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, sub_cost_hv, sub_cost_mv, branch, \
        pop_thresh_lr, line_bc_col, full_ele, wt, coe, grid_ir, \
        grid_om, grid_lifetime = initialization.import_csv_file(step)

    df_weighted = initialization.weighting(df, resolution)

    geo_df, pop_points = initialization.creating_geodataframe(df_weighted, crs,
                                                              unit, input_csv,
                                                              step)
    '-------------------------------------------------------------------------'
    "2.Clustering"

    clustering.sensitivity(pop_points, geo_df)

    geo_df_clustered, clusters_list = clustering.analysis(pop_points, geo_df,
                                                          pop_load)
    '-------------------------------------------------------------------------'
    "3. Grid creation"

    if branch == 'no':
        grid_resume, substations, gdf_roads, roads_segments = \
            grid.routing(geo_df_clustered, geo_df, clusters_list, resolution,
                         pop_thresh, input_sub, line_bc, sub_cost_hv,
                         sub_cost_mv)

        grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                   resolution, line_bc, branch,
                                                   input_sub, gdf_roads,
                                                   roads_segments)

        results.graph(geo_df_clustered, clusters_list, branch, grid_resume_opt,
                      substations)

    elif branch == 'yes':
        gdf_lr = branches.reduce_resolution(input_csv, geo_df, resolution,
                                            geo_df_clustered, clusters_list)

        grid_resume, substations, gdf_roads, roads_segments = \
            branches.routing(geo_df_clustered, geo_df, clusters_list,
                             resolution, pop_thresh, input_sub, line_bc,
                             sub_cost_hv, sub_cost_mv, pop_load, gdf_lr,
                             pop_thresh_lr, line_bc_col, full_ele)

        grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                   resolution, line_bc, branch,
                                                   input_sub, gdf_roads,
                                                   roads_segments)

        results.start_app(geo_df_clustered, clusters_list, branch,
                          grid_resume_opt,
                          substations)
    '-------------------------------------------------------------------------'
    "4.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)

    '-------------------------------------------------------------------------'
    "5.LCOE Analysis"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)
elif step == 2:
    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, sub_cost_hv, sub_cost_mv, branch, \
        pop_thresh_lr, line_bc_col, full_ele, wt, coe, grid_ir, grid_om, \
        grid_lifetime = initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "2.Clustering"

    clustering.sensitivity(pop_points, geo_df)
    geo_df_clustered, clusters_list = \
        clustering.analysis(pop_points, geo_df, pop_load)
    '-------------------------------------------------------------------------'
    "3. Grid creation"

    if branch == 'no':
        grid_resume, substations, gdf_roads, roads_segments = \
            grid.routing(geo_df_clustered, geo_df, clusters_list, resolution,
                         pop_thresh, input_sub, line_bc, sub_cost_hv,
                         sub_cost_mv)

        grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                   resolution, line_bc, branch,
                                                   input_sub, gdf_roads,
                                                   roads_segments)

        results.start_app(geo_df_clustered, clusters_list, branch,
                          grid_resume_opt,
                          substations)

    elif branch == 'yes':
        gdf_lr = branches.reduce_resolution(input_csv, geo_df, resolution,
                                            geo_df_clustered, clusters_list)

        grid_resume, substations, gdf_roads, roads_segments = \
            branches.routing(geo_df_clustered, geo_df, clusters_list,
                             resolution, pop_thresh, input_sub, line_bc,
                             sub_cost_hv, sub_cost_mv, pop_load, gdf_lr,
                             pop_thresh_lr, line_bc_col, full_ele)

        grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                   resolution, line_bc, branch,
                                                   input_sub, gdf_roads,
                                                   roads_segments)

        results.start_app(geo_df_clustered, clusters_list, branch,
                          grid_resume_opt,
                          substations)
    '-------------------------------------------------------------------------'
    "4.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)

    '-------------------------------------------------------------------------'
    "5.LCOE Analysis"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)
elif step == 3:
    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, sub_cost_hv, sub_cost_mv, branch, \
        pop_thresh_lr, line_bc_col, full_ele, wt, coe, grid_ir, grid_om, \
        grid_lifetime, geo_df_clustered, clusters_list = \
        initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "3. Grid creation"

    if branch == 'no':
        grid_resume, substations, gdf_roads, roads_segments = \
            grid.routing(geo_df_clustered, geo_df, clusters_list, resolution,
                         pop_thresh, input_sub, line_bc, sub_cost_hv,
                         sub_cost_mv)

        grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                   resolution, line_bc, branch,
                                                   input_sub, gdf_roads,
                                                   roads_segments)

        results.start_app(geo_df_clustered, clusters_list, branch,
                          grid_resume_opt,
                          substations)

    elif branch == 'yes':
        gdf_lr = branches.reduce_resolution(input_csv, geo_df, resolution,
                                            geo_df_clustered, clusters_list)

        grid_resume, substations, gdf_roads, roads_segments = \
            branches.routing(geo_df_clustered, geo_df, clusters_list,
                             resolution, pop_thresh, input_sub, line_bc,
                             sub_cost_hv, sub_cost_mv, pop_load, gdf_lr,
                             pop_thresh_lr, line_bc_col, full_ele)

        grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                   resolution, line_bc, branch,
                                                   input_sub, gdf_roads,
                                                   roads_segments)

        results.start_app(geo_df_clustered, clusters_list, branch,
                          grid_resume_opt,
                          substations)
    '-------------------------------------------------------------------------'
    "4.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)

    '-------------------------------------------------------------------------'
    "5.Final results"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)

elif step == 4:
    '-------------------------------------------------------------------------'
    "1. Importing Parameters"
    geo_df_clustered, clusters_list, wt, grid_resume, coe, grid_ir, grid_om, \
        grid_lifetime = initialization.import_csv_file(step)

    '-------------------------------------------------------------------------'
    "4.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)

    '-------------------------------------------------------------------------'
    "5.LCOE Analysis"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)
elif step == 5:

    "1. Importing Parameters"
    clusters_list, grid_resume, mg, total_energy, coe, grid_ir, grid_om, \
        grid_lifetime = initialization.import_csv_file(step)
    '-------------------------------------------------------------------------'
    "5.LCOE Analysis"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)
