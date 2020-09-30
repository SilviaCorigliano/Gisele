import os
import dash
import pandas as pd
import geopandas as gpd
import plotly.express as px

import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from Codes import initialization, clustering, processing, collecting

if not os.path.isdir(r'Input'):
    os.chdir('..')

# mg = pd.read_csv(r'Output/Microgrids/microgrids.csv')
mg = pd.DataFrame()
grid_resume_opt = pd.read_csv(r'Output/Grids/grid_resume_opt.csv')
config = pd.read_csv(r'Input/Configuration.csv')
fig = go.Figure(go.Scattermapbox(
    lat=[''],
    lon=[''],
    mode='markers'))

fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Div([
        # html.Img(
        #     src=app.get_asset_url("polimi-logo.png"),
        #     id="polimi-logo",
        #     style={"height": "120px", "width": "auto", 'textAlign': 'center'},
        #     className='four columns'),

        html.H1('GISEle: GIS for Electrification',
                style={'textAlign': 'center', 'color': '000000'}
                ),

        # html.P('Developed by Politecnico di Milano', style={
        #     'textAlign': 'center',
        #     'color': '000000'})
    ]),
    html.Div([dcc.Slider(id='step',
                         min=0,
                         max=4,
                         marks={0: 'Start',
                                1: 'Create GeoDataFrame', 2: 'Clustering',
                                3: 'Grid Routing', 4: 'Microgrid Sizing',
                                5: 'LCOE Analysis'},
                         value=1,
                         )
              ], className='row', style={'textAlign': 'center',
                                         'margin': '20px'}),

        dcc.Tab(label='GIS Configuration', children=[
                    html.Label('Import GIS Data'),
                    dcc.RadioItems(id='import_data',
                                   options=[
                                       {'label': 'Yes', 'value': 'yes'},
                                       {'label': 'No', 'value': 'no'},
                                   ],
                                   value='no',
                                   labelStyle={'display': 'inline-block'}
                                   ),
                    html.Label('Input points .csv file'),
                    dcc.Input(
                        id='input_csv',
                        placeholder='Enter the file name..',
                        debounce=True,
                        type='text',
                        value=''
                    ),
                    html.Label('Input substations .csv file'),
                    dcc.Input(
                        id='input_sub',
                        placeholder='Enter the file name..',
                        debounce=True,
                        type='text',
                        value=''
                    ),
                    html.Label('Coordinate Reference System'),
                    dcc.Input(
                        id='crs',
                        placeholder='Enter the EPSG code..',
                        debounce=True,
                        type='number',
                        value=''
                    ),
                    html.Label('Resolution'),
                    dcc.Input(
                        id='resolution',
                        placeholder='1000m',
                        debounce=True,
                        type='number',
                        min=100, max=4000, step=50,
                        value='1000'
                    ),
        ]),
    html.Div([
        html.Div(id='tabs', children=[

            dcc.Tabs([
                dcc.Tab(label='GIS Configuration', children=[
                    html.Label('Import GIS Data'),
                    dcc.RadioItems(id='import_data',
                                   options=[
                                       {'label': 'Yes', 'value': 'yes'},
                                       {'label': 'No', 'value': 'no'},
                                   ],
                                   value='no',
                                   labelStyle={'display': 'inline-block'}
                                   ),
                    html.Label('Input points .csv file'),
                    dcc.Input(
                        id='input_csv',
                        placeholder='Enter the file name..',
                        debounce=True,
                        type='text',
                        value=''
                    ),
                    html.Label('Input substations .csv file'),
                    dcc.Input(
                        id='input_sub',
                        placeholder='Enter the file name..',
                        debounce=True,
                        type='text',
                        value=''
                    ),
                    html.Label('Coordinate Reference System'),
                    dcc.Input(
                        id='crs',
                        placeholder='Enter the EPSG code..',
                        debounce=True,
                        type='number',
                        value=''
                    ),
                    html.Label('Resolution'),
                    dcc.Input(
                        id='resolution',
                        placeholder='1000m',
                        debounce=True,
                        type='number',
                        min=100, max=4000, step=50,
                        value='1000'
                    ),
                ]),
                dcc.Tab(label='Electrical Configuration',
                        style={'display': 'none'}, children=[
                            html.Label('Population Threshold'),
                            dcc.Input(
                                id='pop_thresh',
                                placeholder='',
                                type='number',
                                min=0, max=1000, step=10,
                                value='100'
                            ),
                            html.Label('Line base cost'),
                            dcc.Input(
                                id='line_bc',
                                placeholder='Enter a value [kW]..',
                                type='number',
                                value=''
                            ),
                            html.Label('Load per capita'),
                            dcc.Input(
                                id='pop_load',
                                placeholder='Enter a value [kW]..',
                                type='number',
                                value=''
                            ),
                            html.Label('HV/MV Substation cost'),
                            dcc.Input(
                                id='sub_cost_HV',
                                placeholder='Enter a value [€]..',
                                type='number',
                                value=''
                            ),
                            html.Label('MV/LV Substation cost'),
                            dcc.Input(
                                id='sub_cost_MV',
                                placeholder='Enter a value [€]..',
                                type='number',
                                value=''
                            ),
                            html.Label('Routing Strategy'),
                            dcc.RadioItems(id='branch',
                                           options=[
                                               {'label': 'Branches',
                                                'value': 'yes'},
                                               {'label': 'Steiner',
                                                'value': 'no'},
                                           ],
                                           value='no',
                                           labelStyle={
                                               'display': 'inline-block'}
                                           ),
                            html.Div(id='branch_options',
                                     style={'display': 'none'},
                                     children=[html.Label(
                                         'Population Treshold (Main Branches)'),
                                         dcc.Input(
                                             id='pop_thresh_lr',
                                             placeholder='Enter a value [kW]..',
                                             type='number',
                                             min=0, max=1000, step=10,
                                             value='200'),

                                         html.Label(
                                             'Line base cost (Collaterals)'),
                                         dcc.Input(
                                             id='line_bc_col',
                                             placeholder='Enter a value [kW]..',
                                             type='number',
                                             value='')]
                                     ),

                            dcc.Checklist(
                                id='full_ele',
                                options=[
                                    {'label': 'Total Electrification',
                                     'value': 'yes'},
                                ],
                                value=['no']
                            )
                        ]),
                dcc.Tab(label='Microgrid Parameters',
                        style={'display': 'none'}, children=[
                    html.Label('Import RES data'),
                    dcc.RadioItems(id='import_res',
                                   options=[
                                       {'label': 'Yes', 'value': 'yes'},
                                       {'label': 'No', 'value': 'no'},
                                   ],
                                   value='yes',
                                   labelStyle={'display': 'inline-block'}
                                   ),
                    html.Label('Cost of Electricity'),
                    dcc.Input(
                        id='coe',
                        placeholder='Enter a value [€/kWh]..',
                        type='number',
                        value=''
                    ),
                    html.Label('Grid Lifetime'),
                    dcc.Input(
                        id='grid_lifetime',
                        placeholder='',
                        min=1, max=100, step=1,
                        type='number',
                        value='40'
                    ),
                    html.Label('Grid O&M Costs'),
                    dcc.Input(
                        id='grid_om',
                        placeholder='[% of total costs]',
                        min=0, max=1, step=0.01,
                        type='number',
                        value='0.01'
                    ),
                    html.Label('Inflation rate'),
                    dcc.Input(
                        id='grid_ir',
                        placeholder='[%/year]',
                        min=0, max=1, step=0.01,
                        type='number',
                        value='0.01'
                    ),
                    html.Label('Wind Turbine Model'),
                    dcc.Dropdown(
                        id='wt',
                        options=[
                            {'label': 'Nordex N27 150',
                             'value': 'Nordex N27 150'},
                            {'label': 'Alstom Eco 80',
                             'value': 'Alstom Eco 80'},
                            {'label': 'Enercon E40 500',
                             'value': 'Enercon E40 500'},
                            {'label': 'Vestas V27 225',
                             'value': 'Vestas V27 225'}
                        ],
                        value='Nordex N27 150'
                    )
                ]),
            ]),
            dcc.Input(
                id='bt_out',
                value='no',
                style={'display': 'none'}),

            html.Div([
                html.Div([html.Button('Run GISEle',
                                      id='bt1', n_clicks=0, disabled=False,
                                      style={'textAlign': 'center',
                                             'margin': '10px'},
                                      className='button-primary')],
                         style={'textAlign': 'center'})
            ]),
            html.P(id='config_out'),
        ], style={'textAlign': 'center'}, className="four columns"),

        html.Div(id='cluster_tab', children=[
            html.Div([
                html.H3('Cluster Sensitivity'),
                html.Label('Limits for the MINIMUM POINTS parameter '),
                dcc.RangeSlider(
                    id='pts',
                    min=0,
                    max=2000,
                    marks={i: '{}'.format(i) for i in range(0, 2200, 200)},
                    step=50,
                    value=[300, 700]
                ),
                html.Label('Limits for the NEIGHBOURHOOD parameter [meters]'),
                dcc.RangeSlider(
                    id='eps',
                    min=0,
                    max=5000,
                    marks={i: '{}'.format(i) for i in range(0, 5500, 500)},
                    step=100,
                    value=[1200, 1700]
                ),
                html.Label('Number of spans'),
                dcc.Input(
                    id='spans',
                    placeholder='',
                    type='number',
                    min=0, max=20, step=1,
                    value='5'),

            ]),
            html.Button(
                'Run Cluster Sensitivity',
                id='bt_cluster_sens', n_clicks=0, disabled=False,
                style={'textAlign': 'center', 'margin': '10px'},
                className='button-primary'),
            html.Div([
                html.H3('Cluster Analysis'),
                html.Label('Chosen value for EPS'),
                dcc.Input(
                    id='eps_final',
                    placeholder='',
                    type='number',
                    min=0, max=10000, step=50,
                    value=''),
                html.Label('Chosen value for minPTS'),
                dcc.Input(
                    id='pts_final',
                    placeholder='',
                    type='number',
                    min=0, max=10000, step=10,
                    value='')
            ]),
            html.Button(
                'Run Cluster Analysis',
                id='bt_cluster', n_clicks=0, disabled=False,
                className='button-primary',
                style={'textAlign': 'center', 'margin': '10px'})
        ], style={'textAlign': 'center'}, className="four columns"),
        html.Div(id='out_tabs_1', children=[
            dcc.Tabs([
                dcc.Tab(label='Topological View', children=[
                    dcc.Graph(
                        id='output_graph',
                        figure=fig,
                    )]),

                dcc.Tab(label='Grid Planning Resume',
                        children=[html.Div(children=[
                            generate_table(grid_resume_opt)])

                        ]),
                dcc.Tab(label='Microgrids',
                        children=[html.Div(children=[
                            generate_table(mg)])
                        ]),
            ]),
        ], style={'textAlign': 'center'}, className='row'),

        html.Div(id='out_tabs_2', children=[
            dcc.Tabs([
                dcc.Tab(label='Sensitivity Graph', children=[
                    dcc.Graph(id='sens_graph')
                ]),
                dcc.Tab(label='Sensitivity Tables',
                        children=[html.Div(id='sens_tables', children=[
                            generate_table(mg)])
                                  ]),
                dcc.Tab(label='Cluster Analysis', children=[
                    dcc.Graph(id='cluster_graph', figure=fig)
                ]),
            ]),
        ], style={'textAlign': 'center'}, className='row')
    ])
])


@app.callback(Output('branch_options', 'style'),
              [Input('branch', 'value')])
def branch_options(branch):
    """ Shows or hides html.Div of the branch options according to the input"""
    if branch == 'yes':
        return {'display': 'inline'}
    else:
        return {'display': 'none'}


@app.callback(Output('bt_out', 'value'),
              [Input('bt1', 'n_clicks'),
               Input('step', 'value')])
def pressed_button(bt1, step):
    """ Checks if the button RUN GISELE was pressed, if yes run the code and
     and changes the bt_out to a value that will call the change_interface
     function """
    button_pressed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'bt1' in button_pressed:
        data_import = config.iloc[0, 1]
        input_csv = config.iloc[1, 1]
        input_sub = config.iloc[2, 1]
        crs = int(config.iloc[3, 1])
        resolution = float(config.iloc[4, 1])

        pop_thresh = float(config.iloc[7, 1])
        line_bc = float(config.iloc[8, 1])
        sub_cost_hv = float(config.iloc[9, 1])
        sub_cost_mv = float(config.iloc[10, 1])
        branch = config.iloc[11, 1]
        pop_thresh_lr = float(config.iloc[12, 1])
        line_bc_col = float(config.iloc[13, 1])
        full_ele = config.iloc[14, 1]
        wt = config.iloc[15, 1]
        coe = float(config.iloc[16, 1])
        grid_ir = float(config.iloc[17, 1])
        grid_om = float(config.iloc[18, 1])
        grid_lifetime = int(config.iloc[19, 1])
        unit = 1
        if step == 1:
            if data_import == 'yes':
                study_area = collecting.data_gathering(crs)
                df = processing.create_mesh(study_area, crs, resolution)
                df_weighted = initialization.weighting(df, resolution)

                geo_df, pop_points = \
                    initialization.creating_geodataframe(df_weighted, crs,
                                                         unit, input_csv, step)
                geo_df.to_file(r"Output/Datasets/geo_df_json",
                               driver='GeoJSON')
                return 'yes'
            df = pd.read_csv(r'Input/' + input_csv + '.csv', sep=',')
            print("Input files successfully imported.")
            df_weighted = initialization.weighting(df, resolution)
            geo_df, pop_points = \
                initialization.creating_geodataframe(df_weighted, crs,
                                                     unit, input_csv, step)
            geo_df.to_file(r"Output/Datasets/geo_df_json", driver='GeoJSON')
            return 'yes'

        elif step > 1:
            df_weighted = \
                pd.read_csv(r'Output/Datasets/' + input_csv + '_weighted.csv')
            valid_fields = ['ID', 'X', 'Y', 'Population', 'Elevation',
                            'Weight']
            blacklist = []
            for x in df_weighted.columns:
                if x not in valid_fields:
                    blacklist.append(x)
            df_weighted.drop(blacklist, axis=1, inplace=True)
            df_weighted.drop_duplicates(['ID'], keep='last', inplace=True)
            print("Input files successfully imported.")
            geo_df, pop_points = \
                initialization.creating_geodataframe(df_weighted, crs,
                                                     unit, input_csv, step)

            geo_df.to_file(r"Output/Datasets/geo_df_json", driver='GeoJSON')
            return 'yes'
    else:
        return 'no'


@app.callback(Output('config_out', 'children'),
              [Input('import_data', 'value'),
               Input('input_csv', 'value'),
               Input('input_sub', 'value'),
               Input('crs', 'value'),
               Input('resolution', 'value'),
               Input('pop_thresh', 'value'),
               Input('line_bc', 'value'),
               Input('pop_load', 'value'),
               Input('sub_cost_HV', 'value'),
               Input('sub_cost_MV', 'value'),
               Input('branch', 'value'),
               Input('pop_thresh_lr', 'value'),
               Input('line_bc_col', 'value'),
               Input('full_ele', 'value'),
               Input('wt', 'value'),
               Input('coe', 'value'),
               Input('grid_ir', 'value'),
               Input('grid_om', 'value'),
               Input('grid_lifetime', 'value')])
def configuration(import_data, input_csv, input_sub, crs, resolution,
                  pop_thresh, line_bc, pop_load, sub_cost_HV, sub_cost_MV,
                  branch, pop_thresh_lr, line_bc_col, full_ele, wt, coe,
                  grid_ir, grid_om, grid_lifetime):
    """ Reads every change of input in the UI and updates the configuration
    file accordingly """
    ctx = dash.callback_context
    if ctx.triggered[0]['value'] is not None:
        para_index = config[config['Parameter'] ==
                            ctx.triggered[0]['prop_id'].split('.')[0]].index

        if isinstance(ctx.triggered[0]['value'], list):
            config.loc[para_index, 'Value'] = ctx.triggered[0]['value'][-1]
        else:
            config.loc[para_index, 'Value'] = ctx.triggered[0]['value']
        msg = 'Parameter changed: ' + str(
            ctx.triggered[0]['prop_id'].split('.')[0])
        print(config)
    else:
        raise PreventUpdate
    return html.Div(msg)


@app.callback(Output('tabs', 'style'),
              [Input('bt_out', 'value')])
def change_interface(bt_out):
    """ Changes the html.Div of the three left tabs to the clustering
     sensitivity and analysis options"""
    if bt_out == 'yes':
        return {'display': 'none'}
    else:
        return {'textAlign': 'center'}


@app.callback(Output('cluster_tab', 'style'),
              [Input('bt_out', 'value')])
def change_interface(bt_out):
    """ Changes the html.Div of the three left tabs to the clustering
     sensitivity and analysis options"""
    if bt_out == 'yes':
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback(Output('out_tabs_1', 'style'),
              [Input('bt_out', 'value')])
def change_interface(bt_out):
    """ Changes the html.Div of the three left tabs to the clustering
     sensitivity and analysis options"""
    if bt_out == 'yes':
        return {'display': 'none'}
    else:
        return {'textAlign': 'center'}


@app.callback(Output('out_tabs_2', 'style'),
              [Input('bt_out', 'value')])
def change_interface(bt_out):
    """ Changes the html.Div of the three left tabs to the clustering
     sensitivity and analysis options"""
    if bt_out == 'yes':
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback(Output('sens_graph', 'figure'),
              [Input('bt_cluster_sens', 'n_clicks'),
               Input('eps', 'value'),
               Input('pts', 'value'),
               Input('spans', 'value')])
def cluster_sensitivity(bt_cluster_sens, eps, pts, spans):
    """ Checks if the button RUN GISELE was pressed, if yes run the code and
     and changes the bt_out to a value that will call the change_interface
     function """
    button_pressed = [p['prop_id'] for p in dash.callback_context.triggered][0]

    if 'bt_cluster_sens' in button_pressed:
        geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
        loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
        pop_points = pd.DataFrame(data=loc).values
        fig_sens = clustering.sensitivity2(pop_points, geo_df, eps, pts,
                                           int(spans))
        return fig_sens
    return fig


@app.callback(Output('cluster_graph', 'figure'),
              [Input('bt_cluster', 'n_clicks'),
               Input('eps_final', 'value'),
               Input('pts_final', 'value')])
def cluster_sensitivity(bt_cluster, eps_final, pts_final):
    """ Checks if the button RUN GISELE was pressed, if yes run the code and
     and changes the bt_out to a value that will call the change_interface
     function """
    button_pressed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    pop_load = float(config.iloc[6, 1])
    if 'bt_cluster' in button_pressed:
        geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
        loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
        pop_points = pd.DataFrame(data=loc).values
        geo_df_clustered, clusters_list, fig_clusters = \
            clustering.analysis2(pop_points, geo_df, pop_load,
                                 eps_final, pts_final)
        return fig_clusters
    return fig


app.run_server(debug=True)
