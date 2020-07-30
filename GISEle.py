"""
GIS For Electrification (GISEle)
Developed by the Energy Department of Politecnico di Milano
Running Code

In order to run the following algorithm please check if the Input files are
configured accordingly, for more information check the README.
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
         "4.Main branch and collateral's",
         '5.Microgrid Sizing',
         '6.LCOE Analysis']
print("\n".join(steps))
supporting_GISEle2.s()
step = int(input('Which step do you want to select?: '))
# step = 6
if step == 1:
    '-------------------------------------------------------------------------'
    "1. Assigning weights, Cleaning and Creating the GeoDataFrame"

    df, input_sub, input_csv, crs, resolution, unit, pop_load, pop_thresh, \
        line_bc, limit_hv, limit_mv, wt, coe, grid_ir, grid_om, grid_lifetime\
        = initialization.import_csv_file(step)

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

    grid_resume, substations = grid.routing(geo_df_clustered, geo_df,
                                            clusters_list, resolution,
                                            pop_thresh, input_sub, line_bc,
                                            limit_hv, limit_mv)

    grid_resume_opt = optimization.connections(geo_df, grid_resume, resolution,
                                               line_bc, limit_hv, limit_mv,
                                               step)

    results.graph(geo_df_clustered, clusters_list, step, grid_resume_opt,
                  substations)
    '-------------------------------------------------------------------------'
    "5.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)

    '-------------------------------------------------------------------------'
    "6.LCOE Analysis"

    # final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
    #                                                total_energy,
    #                                                grid_resume, mg_NPC)

elif step == 2:
    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv, wt, coe, grid_ir, grid_om, \
        grid_lifetime = initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "2.Clustering"

    clustering.sensitivity(pop_points, geo_df)
    geo_df_clustered, clusters_list = \
        clustering.analysis(pop_points, geo_df, pop_load)
    '-------------------------------------------------------------------------'
    "3.Grid creation"
    grid_resume, substations = grid.routing(geo_df_clustered, geo_df,
                                            clusters_list, resolution,
                                            pop_thresh, input_sub, line_bc,
                                            limit_hv, limit_mv)

    grid_resume_opt = optimization.connections(geo_df, grid_resume, resolution,
                                               line_bc, limit_hv, limit_mv,
                                               step)

    results.graph(geo_df_clustered, clusters_list, step, grid_resume_opt,
                  substations)

    '-------------------------------------------------------------------------'
    "5.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)


elif step == 3:
    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv, geo_df_clustered, \
        clusters_list, wt, coe, grid_ir, grid_om, grid_lifetime \
        = initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "3. Grid creation"

    grid_resume, substations = grid.routing(geo_df_clustered, geo_df,
                                            clusters_list, resolution,
                                            pop_thresh, input_sub, line_bc,
                                            limit_hv, limit_mv)

    grid_resume_opt = optimization.connections(geo_df, grid_resume, resolution,
                                               line_bc, limit_hv, limit_mv,
                                               step)

    results.graph(geo_df_clustered, clusters_list, step, grid_resume_opt,
                  substations)
    '-------------------------------------------------------------------------'
    "5.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)

    '-------------------------------------------------------------------------'
    "6.Final results"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)


elif step == 4:

    '-------------------------------------------------------------------------'
    "1. Importing and Creating the GeoDataFrame"

    df_weighted, input_sub, input_csv, crs, resolution, unit, pop_load, \
        pop_thresh, line_bc, limit_hv, limit_mv, geo_df_clustered, \
        clusters_list, input_csv_lr, pop_thresh_lr, line_bc_col, \
        full_ele, wt, coe, grid_ir, grid_om, grid_lifetime \
        = initialization.import_csv_file(step)

    geo_df, pop_points = initialization. \
        creating_geodataframe(df_weighted, crs, unit, input_csv, step)

    '-------------------------------------------------------------------------'
    "4.Main branch and collateral's"
    grid_resume, substations = branches.routing(geo_df_clustered, geo_df,
                                                clusters_list, resolution,
                                                pop_thresh, input_sub, line_bc,
                                                limit_hv, limit_mv, pop_load,
                                                input_csv_lr, pop_thresh_lr,
                                                line_bc_col, full_ele)

    grid_resume_opt = optimization.connections(geo_df, grid_resume, resolution,
                                               line_bc, limit_hv, limit_mv,
                                               step)

    results.graph(geo_df_clustered, clusters_list, step, grid_resume_opt,
                  substations)

    '-------------------------------------------------------------------------'
    "5.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)
    '-------------------------------------------------------------------------'
    "6.LCOE Analysis"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)

elif step == 5:
    '-------------------------------------------------------------------------'
    "1. Importing Parameters"
    geo_df_clustered, clusters_list, wt, grid_resume, coe, grid_ir, grid_om, \
        grid_lifetime = initialization.import_csv_file(step)

    '-------------------------------------------------------------------------'
    "5.Microgrid sizing"
    load_profile, years, total_energy = supporting_GISEle2.load(clusters_list,
                                                                grid_lifetime)

    mg = supporting_GISEle2.sizing(load_profile, clusters_list,
                                   geo_df_clustered, wt, years)

    '-------------------------------------------------------------------------'
    "6.LCOE Analysis"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)
elif step == 6:

    "1. Importing Parameters"
    clusters_list, grid_resume, mg, total_energy, coe, grid_ir, grid_om, \
        grid_lifetime = initialization.import_csv_file(step)
    '-------------------------------------------------------------------------'
    "6.LCOE Analysis"

    final_LCOEs = supporting_GISEle2.lcoe_analysis(clusters_list,
                                                   total_energy,
                                                   grid_resume, mg, coe,
                                                   grid_ir, grid_om,
                                                   grid_lifetime)