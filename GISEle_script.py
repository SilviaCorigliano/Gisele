import os

import base64
import io
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import plotly.graph_objs as go
from shapely.geometry import Point
from functions import load, sizing
from gisele import initialization, clustering, processing, collecting, \
    optimization, results, grid, branches
import pyutilib.subprocess.GlobalData

pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

# creation of all global variables used in the algorithm
gis_columns = pd.DataFrame(columns=['ID', 'X', 'Y', 'Population', 'Elevation',
                                    'Weight'])

mg_columns = pd.DataFrame(columns=['Cluster', 'PV [kW]', 'Wind [kW]',
                                   'Diesel [kW]', 'BESS [kWh]',
                                   'Inverter [kW]', 'Investment Cost [k€]',
                                   'OM Cost [k€]', 'Replace Cost [k€]',
                                   'Total Cost [k€]', 'Energy Produced [MWh]',
                                   'Energy consumed [MWh]', 'LCOE [€/kWh]'])

lcoe_columns = pd.DataFrame(columns=['Cluster', 'Grid NPC [k€]', 'MG NPC [k€]',
                                     'Grid Energy Consumption [MWh]',
                                     'MG LCOE [€/kWh]',
                                     'Grid LCOE [€/kWh]', 'Best Solution'])

# breaking load profile in half for proper showing in the data table
input_profile = pd.read_csv(r'Input/Load Profile.csv').round(4)
load_profile = pd.DataFrame(columns=['Hour 0-12', 'Power [p.u.]',
                                     'Hour 12-24', 'Power (p.u.)'])
load_profile['Hour 0-12'] = pd.Series(np.arange(12)).astype(int)
load_profile['Hour 12-24'] = pd.Series(np.arange(12, 24)).astype(int)
load_profile['Power [p.u.]'] = input_profile.iloc[0:12, 0].values
load_profile['Power (p.u.)'] = input_profile.iloc[12:24, 0].values
lp_data = load_profile.to_dict('records')

# configuration file, eps and pts values separated by - since they are a range
config = pd.read_csv(r'Input/Configuration.csv')
config.loc[21, 'Value'] = sorted(list(map(int,
                                          config.loc[21, 'Value'].split('-'))))
config.loc[20, 'Value'] = sorted(list(map(int,
                                          config.loc[20, 'Value'].split('-'))))

# empty map to show as initial output
fig = go.Figure(go.Scattermapbox(
    lat=[''],
    lon=[''],
    mode='markers'))

fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

# study area definition
lon_point_list = [-65.07295675004093, -64.75263629329811, -64.73311537903679,
                  -65.06630189290638]
lat_point_list = [-17.880592240953966, -17.86581365258916, -18.065431449408248,
                  -18.07471050015602]
polygon_geom = Polygon(zip(lon_point_list, lat_point_list))
study_area = gpd.GeoDataFrame(index=[0], crs=4326,
                              geometry=[polygon_geom])


"""
STEPS
{ 1: 'GIS Data Processing', 2: 'Clustering',3: 'Grid Routing', 
4: 'Microgrid Sizing',5: 'Clusters interconnections' 6: 'LCOE Analysis'},
"""
step = 3

if step <2:
    '-------------------------------------------------------------------------'
    "1. Assigning weights, Cleaning and Creating the GeoDataFrame"

# decide whether to import population data from csv ('yes or 'no')
    import_pop_value = 'no'

    data_import = config.iloc[0, 1]
    input_csv = config.iloc[1, 1]
    crs = int(config.iloc[3, 1])
    resolution = float(config.iloc[4, 1])
    landcover_option = (config.iloc[27, 1])
    unit = 1
    step = 1

    if data_import == 'yes':
        collecting.data_gathering(crs, study_area)
        landcover_option = 'CGLS'
        if not import_pop_value:
            df = processing.create_mesh(study_area, crs, resolution)
        else:
            imported_pop = pd.read_csv(r'Input/imported_pop.csv')
            df = processing.create_mesh(study_area, crs, resolution,
                                        imported_pop)
        df_weighted = initialization.weighting(df, resolution,
                                               landcover_option)

        geo_df, pop_points = \
            initialization.creating_geodataframe(df_weighted, crs,
                                                 unit, input_csv, step)
        geo_df.to_file(r"Output/Datasets/geo_df_json",
                       driver='GeoJSON')
    else:
        df = pd.read_csv(r'Input/' + input_csv + '.csv', sep=',')
        print("Input files successfully imported.")
        df_weighted = initialization.weighting(df, resolution,
                                               landcover_option)
        geo_df, pop_points = \
            initialization.creating_geodataframe(df_weighted, crs,
                                                 unit, input_csv, step)
        geo_df.to_file(r"Output/Datasets/geo_df_json", driver='GeoJSON')
        initialization.roads_import(geo_df, crs)
    geo_df = geo_df.to_crs(epsg=4326)
    fig2 = go.Figure(go.Scattermapbox(

        name='GeoDataFrame',
        lat=geo_df.geometry.y,
        lon=geo_df.geometry.x,
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=10,
            showscale=True,
            color=geo_df.Population,
            opacity=0.8,
            colorbar=dict(title='People')
        ),
        text=list(
            zip(geo_df.ID, geo_df.Weight, geo_df.Population.round(1))),
        hoverinfo='text',
        below="''"
    ))
    fig2.update_layout(mapbox_style="carto-positron")
    fig2.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig2.update_layout(mapbox_zoom=8.5,
                       mapbox_center={"lat": geo_df.geometry.y[
                           (int(geo_df.shape[0] / 2))],
                                      "lon": geo_df.geometry.x[
                                          (int(geo_df.shape[0] / 2))]})

    fig2.show()
#todo->insert choices and options for importing data automatically
    '-------------------------------------------------------------------------'
if step <3:
    "2.Clustering"
#decide wether to perform sensitivity analysis ('yes' or 'no')
    sensitivity='no'
    if sensitivity=='yes':
        resolution = float(config.iloc[4, 1])
        eps = list(config.iloc[20, 1])
        pts = list(config.iloc[21, 1])
        spans = int(config.iloc[22, 1])

        geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
        loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
        pop_points = pd.DataFrame(data=loc).values
        fig_sens = clustering.sensitivity(resolution, pop_points, geo_df, eps,
                                          pts,
                                          int(spans))
        fig_sens.show()

    eps_final = int(config.iloc[23, 1])
    pts_final = int(config.iloc[24, 1])
    c1_merge = int(config.iloc[25, 1])
    c2_merge = int(config.iloc[26, 1])
    pop_load = float(config.iloc[6, 1])
    geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
    loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
    pop_points = pd.DataFrame(data=loc).values

    geo_df_clustered, clusters_list = \
        clustering.analysis(pop_points, geo_df, pop_load,
                            eps_final, pts_final)

    fig_clusters = clustering.plot_clusters(geo_df_clustered,
                                            clusters_list)
    clusters_list.to_csv(r"Output/Clusters/clusters_list.csv",
                         index=False)
    geo_df_clustered.to_file(r"Output/Clusters/geo_df_clustered.json",
                             driver='GeoJSON')
    fig_clusters.show()

# todo->insert cluster merging options
    '-------------------------------------------------------------------------'
if step <4:
    "3. Grid creation"

    input_csv = config.iloc[1, 1]
    input_sub = config.iloc[2, 1]
    resolution = float(config.iloc[4, 1])
    pop_load = float(config.iloc[6, 1])
    pop_thresh = float(config.iloc[7, 1])
    line_bc = float(config.iloc[8, 1])
    sub_cost_hv = float(config.iloc[9, 1])
    sub_cost_mv = float(config.iloc[10, 1])
    branch = config.iloc[11, 1]
    pop_thresh_lr = float(config.iloc[12, 1])
    line_bc_col = float(config.iloc[13, 1])
    full_ele = config.iloc[14, 1]
    geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
    geo_df_clustered = \
        gpd.read_file(r"Output/Clusters/geo_df_clustered.json")
    clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
    clusters_list.index = clusters_list.Cluster.values

    if branch == 'no':
        grid_resume, substations, gdf_roads, roads_segments = \
            grid.routing(geo_df_clustered, geo_df, clusters_list,
                         resolution, pop_thresh, input_sub, line_bc,
                         sub_cost_hv, sub_cost_mv, full_ele)


        fig_grid = results.graph(geo_df_clustered, clusters_list, branch,
                                 grid_resume, substations, pop_thresh,
                                 full_ele)

    elif branch == 'yes':
        gdf_lr = branches.reduce_resolution(input_csv, geo_df, resolution,
                                            geo_df_clustered,
                                            clusters_list)

        grid_resume, substations, gdf_roads, roads_segments = \
            branches.routing(geo_df_clustered, geo_df, clusters_list,
                             resolution, pop_thresh, input_sub, line_bc,
                             sub_cost_hv, sub_cost_mv, pop_load, gdf_lr,
                             pop_thresh_lr, line_bc_col, full_ele)

        fig_grid = results.graph(geo_df_clustered, clusters_list, branch,
                                 grid_resume, substations, pop_thresh,
                                 full_ele)
    fig_grid.show()
'-------------------------------------------------------------------------'
if step <5:
    "4.Microgrid sizing"
    grid_lifetime = int(config.iloc[19, 1])
    wt = (config.iloc[15, 1])

    geo_df_clustered = \
        gpd.read_file(r"Output/Clusters/geo_df_clustered.json")

    clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
    clusters_list.index = clusters_list.Cluster.values

    yearly_profile, years, total_energy = load(clusters_list,
                                               grid_lifetime,
                                               input_profile)

    mg = sizing(yearly_profile, clusters_list, geo_df_clustered, wt, years)
    fig_mg = results.graph_mg(mg, geo_df_clustered, clusters_list)
    fig_mg.show()

    '-------------------------------------------------------------------------'
if step <6:
    "5.MILP input preparations"

    branch = config.iloc[11, 1]
    full_ele = config.iloc[14, 1]
    input_sub = config.iloc[2, 1]
    pop_thresh = float(config.iloc[7, 1])
    coe = float(config.iloc[16, 1])
    grid_om = float(config.iloc[18, 1])
    grid_ir = float(config.iloc[17, 1])
    grid_lifetime = int(config.iloc[19, 1])
    resolution = float(config.iloc[4, 1])
    line_bc = float(config.iloc[8, 1])

    geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
    geo_df_clustered = \
        gpd.read_file(r"Output/Clusters/geo_df_clustered.json")
    clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
    clusters_list.index = clusters_list.Cluster.values
    substations = pd.read_csv(r'Input/' + input_sub + '.csv')
    geometry = [Point(xy) for xy in
                zip(substations['X'], substations['Y'])]
    substations = gpd.GeoDataFrame(substations, geometry=geometry,
                                   crs=geo_df.crs)

    mg = pd.read_csv('Output/Microgrids/microgrids.csv')
    mg.index = mg.Cluster.values
    total_energy = pd.read_csv('Output/Microgrids/Grid_energy.csv')
    total_energy.index = total_energy.Cluster.values

    if branch == 'yes':
        grid_resume = pd.read_csv(r'Output/Branches/grid_resume.csv')
        grid_resume.index = grid_resume.Cluster.values

    else:
        grid_resume = pd.read_csv(r'Output/Grids/grid_resume.csv')
        grid_resume.index = grid_resume.Cluster.values


    grid_resume_opt = \
        optimization.clusters_interconnections(geo_df_clustered, grid_resume,
                               substations, mg, total_energy, grid_om, coe,
                               grid_ir, grid_lifetime, branch, line_bc,
                               resolution)

    '-------------------------------------------------------------------------'
if step < 7:
    "6.LCOE Analysis"
#define p max along MV lines
    p_max_lines =10000
    branch = config.iloc[11, 1]
    full_ele = config.iloc[14, 1]
    input_sub = config.iloc[2, 1]
    pop_thresh = float(config.iloc[7, 1])
    coe = float(config.iloc[16, 1])
    grid_om = float(config.iloc[18, 1])
    grid_ir = float(config.iloc[17, 1])
    grid_lifetime = int(config.iloc[19, 1])
    resolution = float(config.iloc[4, 1])
    line_bc = float(config.iloc[8, 1])

    geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
    geo_df_clustered = \
        gpd.read_file(r"Output/Clusters/geo_df_clustered.json")
    clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
    clusters_list.index = clusters_list.Cluster.values
    substations = pd.read_csv(r'Input/' + input_sub + '.csv')
    geometry = [Point(xy) for xy in
                zip(substations['X'], substations['Y'])]
    substations = gpd.GeoDataFrame(substations, geometry=geometry,
                                   crs=geo_df.crs)

    mg = pd.read_csv('Output/Microgrids/microgrids.csv')
    mg.index = mg.Cluster.values
    total_energy = pd.read_csv('Output/Microgrids/Grid_energy.csv')
    total_energy.index = total_energy.Cluster.values

    if branch == 'yes':
        grid_resume = pd.read_csv(r'Output/Branches/grid_resume.csv')
        grid_resume.index = grid_resume.Cluster.values

    else:
        grid_resume = pd.read_csv(r'Output/Grids/grid_resume.csv')
        grid_resume.index = grid_resume.Cluster.values


    grid_resume_opt = \
        optimization.milp_execution(geo_df_clustered, grid_resume, substations, coe, branch, line_bc,
              resolution,p_max_lines)

    fig_grid = results.graph(geo_df_clustered, clusters_list, branch,
                             grid_resume_opt, substations, pop_thresh,
                             full_ele)
    if branch == 'yes':
        file = 'Output/Branches/all_connections_opt'
        if os.path.isfile(file + '.shp'):
            results.line_break(file, fig_grid, 'black')

    else:
        file = 'Output/Grids/all_connections_opt'
        if os.path.isfile(file + '.shp'):
            results.line_break(file, fig_grid, 'black')

fig_grid.show()





# @app.callback(Output('upload_pop_out', 'children'),
#               [Input('upload_pop', 'contents')])
# def read_upload_pop(contents):
#     if contents is not None:
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)
#         df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#         df.to_csv(r'Input/imported_pop.csv', index=False)
#     return '_'
#
#
# @app.callback(Output('upload_csv_out', 'children'),
#               [Input('upload_csv', 'contents')],
#               [State('upload_csv', 'filename')])
# def read_upload_csv(contents, filename):
#     if dash.callback_context.triggered:
#         if contents is not None:
#             if not 'csv' in filename:
#                 return html.P('The selected file is not a CSV table.')
#             content_type, content_string = contents.split(',')
#             decoded = base64.b64decode(content_string)
#             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#             df.to_csv(r'Input/imported_csv.csv', index=False)
#             return html.P('CSV file successfully imported.')
#         return html.P('No files selected.')
#     return html.P('No files selected.')
#
#
# @app.callback(Output('upload_subs_out', 'children'),
#               [Input('upload_subs', 'contents')],
#               [State('upload_subs', 'filename')])
# def read_upload_csv(contents, filename):
#     if dash.callback_context.triggered:
#         if contents is not None:
#             if not 'csv' in filename:
#                 return html.P('The selected file is not a CSV table.')
#             content_type, content_string = contents.split(',')
#             decoded = base64.b64decode(content_string)
#             df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
#             df.to_csv(r'Input/imported_subs.csv', index=False)
#             return html.P('CSV file successfully imported.')
#         return html.P('No files selected.')
#     return html.P('No files selected.')
#
#
# @app.callback(Output('studyarea_out', 'children'),
#               [Input('lat1', 'value'), Input('lat2', 'value'),
#                Input('lat3', 'value'), Input('lat4', 'value'),
#                Input('lon1', 'value'), Input('lon2', 'value'),
#                Input('lon3', 'value'), Input('lon4', 'value')])
# def create_study_area(lat1, lat2, lat3, lat4, lon1, lon2, lon3, lon4):
#     if dash.callback_context.triggered:
#         lat_points = [lat1, lat2, lat3, lat4]
#         lon_points = [lon1, lon2, lon3, lon4]
#         study_area.geometry = [Polygon(zip(lon_points, lat_points))]
#         study_area.to_file(r'Input/study_area.shp')
#     return '_'









# @app.callback(Output('output_cluster', 'figure'),
#               [Input('cluster_analysis', 'n_clicks'),
#                Input('bt_merge_clusters', 'n_clicks')])
# def analysis(cluster_analysis, bt_merge_clusters):
#     """ Checks if the button RUN GISELE was pressed, if yes run the code and
#      and changes the bt_out to a value that will call the change_interface
#      function """
#     eps_final = int(config.iloc[23, 1])
#     pts_final = int(config.iloc[24, 1])
#     c1_merge = int(config.iloc[25, 1])
#     c2_merge = int(config.iloc[26, 1])
#     button_pressed = [p['prop_id'] for p in dash.callback_context.triggered][0]
#     pop_load = float(config.iloc[6, 1])
#     if 'cluster_analysis' in button_pressed:
#         geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
#         loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
#         pop_points = pd.DataFrame(data=loc).values
#
#         geo_df_clustered, clusters_list = \
#             clustering.analysis(pop_points, geo_df, pop_load,
#                                 eps_final, pts_final)
#
#         fig_clusters = clustering.plot_clusters(geo_df_clustered,
#                                                 clusters_list)
#         clusters_list.to_csv(r"Output/Clusters/clusters_list.csv",
#                              index=False)
#         geo_df_clustered.to_file(r"Output/Clusters/geo_df_clustered.json",
#                                  driver='GeoJSON')
#         return fig_clusters
#
#     if 'bt_merge_clusters' in button_pressed:
#         geo_df_clustered = \
#             gpd.read_file(r"Output/Clusters/geo_df_clustered.json")
#         geo_df_clustered.loc[geo_df_clustered['Cluster'] ==
#                              c2_merge, 'Cluster'] = c1_merge
#
#         clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
#         drop_index = \
#             clusters_list.index[clusters_list['Cluster'] == c2_merge][0]
#         clusters_list = clusters_list.drop(index=drop_index)
#
#         fig_merged = clustering.plot_clusters(geo_df_clustered,
#                                               clusters_list)
#
#         clusters_list.to_csv(r"Output/Clusters/clusters_list.csv",
#                              index=False)
#         geo_df_clustered.to_file(r"Output/Clusters/geo_df_clustered.json",
#                                  driver='GeoJSON')
#         return fig_merged
#     raise PreventUpdate




# @app.callback(Output('datatable_gis', 'data'),
#               [Input('datatable_gis', "page_current"),
#                Input('datatable_gis', "page_size"),
#                Input('output_gis', "figure")])
# def update_table(page_current, page_size, output_gis):
#     if os.path.isfile(r'Output/Datasets/geo_df_json'):
#         geo_df2 = pd.DataFrame(
#             gpd.read_file(r"Output/Datasets/geo_df_json").drop(
#                 columns='geometry'))
#     else:
#         geo_df2 = pd.DataFrame()
#     return geo_df2.iloc[
#            page_current * page_size:(page_current + 1) * page_size
#            ].to_dict('records')
#
#
# @app.callback([Output('datatable_grid', 'data'),
#                Output('datatable_grid', 'columns')],
#               [Input('datatable_grid', "page_current"),
#                Input('datatable_grid', "page_size"),
#                Input('datatable_grid', 'sort_by'),
#                Input('output_grid', "figure")],
#               [State('branch', 'value')])
# def update_table(page_current, page_size, sort_by, output_grid, branches):
#     branch = config.iloc[11, 1]
#     geo_df2 = pd.DataFrame()
#
#     if branch == 'no':
#         if os.path.isfile(r'Output/Grids/grid_resume.csv'):
#             geo_df2 = pd.read_csv(r'Output/Grids/grid_resume.csv')
#             geo_df2 = geo_df2.round(2)
#
#             if len(sort_by):
#                 geo_df2 = geo_df2.sort_values(
#                     sort_by[0]['column_id'],
#                     ascending=sort_by[0]['direction'] == 'asc',
#                     inplace=False)
#
#     if branch == 'yes':
#         if os.path.isfile(r'Output/Branches/grid_resume.csv'):
#             geo_df2 = pd.read_csv(r'Output/Branches/grid_resume.csv')
#             geo_df2 = geo_df2.round(2)
#
#             if len(sort_by):
#                 geo_df2 = geo_df2.sort_values(
#                     sort_by[0]['column_id'],
#                     ascending=sort_by[0]['direction'] == 'asc',
#                     inplace=False)
#     geo_df2 = geo_df2.dropna(axis=1, how='all')
#     columns = [{"name": i, "id": i} for i in geo_df2.columns]
#     return geo_df2.iloc[
#            page_current * page_size:(page_current + 1) * page_size
#            ].to_dict('records'), columns
#
#
# @app.callback(Output('datatable_mg', 'data'),
#               [Input('datatable_mg', "page_current"),
#                Input('datatable_mg', "page_size"),
#                Input('datatable_mg', 'sort_by'),
#                Input('output_mg', "figure")])
# def update_table(page_current, page_size, sort_by, output_mg):
#     if os.path.isfile(r'Output/Microgrids/microgrids.csv'):
#         geo_df2 = pd.read_csv(r'Output/Microgrids/microgrids.csv')
#         geo_df2 = geo_df2.round(2)
#
#         if len(sort_by):
#             geo_df2 = geo_df2.sort_values(
#                 sort_by[0]['column_id'],
#                 ascending=sort_by[0]['direction'] == 'asc',
#                 inplace=False)
#     else:
#         geo_df2 = pd.DataFrame()
#     return geo_df2.iloc[
#            page_current * page_size:(page_current + 1) * page_size
#            ].to_dict('records')
#
#
# @app.callback(Output('load_profile_graph', 'figure'),
#               [Input('datatable_load_profile', 'data'),
#                Input('output_mg', "figure")])
# def update_table(datatable_load_profile, output_mg):
#     lp = pd.DataFrame(datatable_load_profile)
#     graph = go.Figure(data=go.Scatter(x=np.arange(23),
#                                       y=lp['Power [p.u.]'].append(
#                                           lp['Power (p.u.)']),
#                                       mode='lines+markers'))
#     graph.update_layout(title='Daily Load Profile',
#                         xaxis_title='Hour',
#                         yaxis_title='Power [p.u.]')
#     return graph
#
#
#
# @app.callback([Output('datatable_grid_final', 'data'),
#                Output('datatable_grid_final', 'columns')],
#               [Input('datatable_grid_final', "page_current"),
#                Input('datatable_grid_final', "page_size"),
#                Input('datatable_grid_final', 'sort_by'),
#                Input('output_lcoe', "figure")],
#               [State('branch', 'value')])
# def update_table(page_current, page_size, sort_by, output_grid, branches):
#     branch = config.iloc[11, 1]
#     geo_df2 = pd.DataFrame()
#
#     if branch == 'no':
#         if os.path.isfile(r'Output/Grids/grid_resume_opt.csv'):
#             geo_df2 = pd.read_csv(r'Output/Grids/grid_resume_opt.csv')
#             geo_df2 = geo_df2.round(2)
#
#             if len(sort_by):
#                 geo_df2 = geo_df2.sort_values(
#                     sort_by[0]['column_id'],
#                     ascending=sort_by[0]['direction'] == 'asc',
#                     inplace=False)
#
#     if branch == 'yes':
#         if os.path.isfile(r'Output/Branches/grid_resume_opt.csv'):
#             geo_df2 = pd.read_csv(r'Output/Branches/grid_resume_opt.csv')
#             geo_df2 = geo_df2.round(2)
#
#             if len(sort_by):
#                 geo_df2 = geo_df2.sort_values(
#                     sort_by[0]['column_id'],
#                     ascending=sort_by[0]['direction'] == 'asc',
#                     inplace=False)
#     geo_df2 = geo_df2.dropna(axis=1, how='all')
#     columns = [{"name": i, "id": i} for i in geo_df2.columns]
#     return geo_df2.iloc[
#            page_current * page_size:(page_current + 1) * page_size
#            ].to_dict('records'), columns
#
#
# @app.callback(Output('datatable_load_profile', 'data'),
#               [Input('upload_loadprofile', 'contents')],
#               [State('upload_loadprofile', 'filename'),
#                State('upload_loadprofile', 'last_modified')])
# def update_output(contents, filename, last_modified):
#     if contents is not None:
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)
#         df = pd.read_csv(io.StringIO(decoded.decode('utf-8'))).round(4)
#         upload_profile = pd.DataFrame(columns=['Hour 0-12', 'Power [p.u.]',
#                                                'Hour 12-24',
#                                                'Power (p.u.)'])
#         upload_profile['Hour 0-12'] = pd.Series(np.arange(12)).astype(int)
#         upload_profile['Hour 12-24'] = pd.Series(np.arange(12, 24)).astype(
#             int)
#         upload_profile['Power [p.u.]'] = df.iloc[0:12, 0].values
#         upload_profile['Power (p.u.)'] = df.iloc[12:24, 0].values
#
#         return upload_profile.to_dict('records')
#     raise PreventUpdate


