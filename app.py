import os
import dash
import pandas as pd
import geopandas as gpd
import dash_table
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from shapely.geometry import Point
from supporting_GISEle2 import load, sizing, lcoe_analysis
from Codes import initialization, clustering, processing, collecting, \
    optimization, results, grid, branches
import pyutilib.subprocess.GlobalData

pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

gis_columns = pd.DataFrame(columns=['ID', 'X', 'Y', 'Population', 'Elevation',
                                    'Weight'])

mg_columns = pd.DataFrame(columns=['Cluster', 'PV' '[kW]', 'Wind [kW]',
                                   'Diesel [kW]', 'BESS [kWh]',
                                   'Inverter [kW]', 'Investment Cost [k€]',
                                   'OM Cost [k€]', 'Replace Cost [k€]',
                                   'Total Cost [k€]', 'Energy Produced [MWh]',
                                   'Energy Consumed [MWh]', 'LCOE [€/kWh]'])

lcoe_columns = pd.DataFrame(columns=['Cluster', 'Grid NPC [k€]', 'MG NPC [k€]',
                                     'Grid Energy Consumption [MWh]',
                                     'MG LCOE [€/kWh]',
                                     'Grid LCOE [€/kWh]', 'Best Solution'])

config = pd.read_csv(r'Input/Configuration3.csv')
config.loc[21, 'Value'] = sorted(list(map(int,
                                          config.loc[21, 'Value'].split('-'))))
config.loc[20, 'Value'] = sorted(list(map(int,
                                          config.loc[20, 'Value'].split('-'))))
fig = go.Figure(go.Scattermapbox(
    lat=[''],
    lon=[''],
    mode='markers'))

fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
app.layout = html.Div([
    html.Div([dbc.Row([
        dbc.Col(
            html.Img(
                src=app.get_asset_url("poli_horizontal.png"),
                id="polimi-logo",
                style={"height": "80px", "width": "auto",
                       'textAlign': 'center'},
                className='four columns'),
            width={"size": 2, "offset": 1}, align='center'),
        dbc.Col(
            html.H1('GISEle: GIS for Electrification',
                    style={'textAlign': 'center', 'color': '000000'}
                    ),
            width={"size": 5, "offset": 2}, align='center')

    ], ),
    ]),
    html.Div([dcc.Slider(id='step',
                         min=0,
                         max=5,
                         marks={0: 'Start',
                                1: 'GIS Data Processing', 2: 'Clustering',
                                3: 'Grid Routing', 4: 'Microgrid Sizing',
                                5: 'LCOE Analysis'},
                         value=1,
                         )
              ], style={'textAlign': 'center', 'margin': '20px'}),

    html.Div(id='start_div', children=[
        dbc.Row([
            dbc.Col(html.Div(style={'textAlign': 'center'}, children=[
                html.H2(['Introduction'], style={'margin': '20px',
                                                 'color': "#55b298"}),
                dbc.Card([
                    dbc.CardBody(["The GIS for electrification (GISEle) "
                                  "is an open source Python-based tool that "
                                  "uses GIS and terrain analysis to model the "
                                  "area under study, groups loads using a "
                                  "density-based "
                                  "clustering algorithm called DBSCAN and "
                                  "then it uses graph theory to find the "
                                  "least-costly electric network topology "
                                  "that can connect all the people in the "
                                  "area."], style={'textAlign': 'justify'}),
                    dbc.CardFooter(
                        dbc.CardLink("Energy4Growing",
                                     href="http://www.e4g.polimi.it/")),
                ], className="mb-3", style={}),

                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            'NEXT',
                            size='lg',
                            color='primary',
                            id='next_step_start', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 6}, align='center'),
                ], justify='around'),
            ]),
                    width={"size": 3, "offset": 0}),

            dbc.Col(html.Div([
                dcc.Graph(
                    id='output_start',
                    figure=fig,
                )
            ]), width={"size": 8, "offset": 0}, align='center'),

        ], justify="around"),
    ]),

    html.Div(id='gis_div', children=[
        dbc.Row([
            dbc.Col(html.Div(style={'textAlign': 'center'}, children=[
                html.H2(['GIS Data Analysis'], style={'color': "#55b298"}),
                dbc.Checklist(id='import_data', switch=True, inline=True,
                              options=[
                                  {'label': 'Import GIS Data',
                                   'value': 'y'},
                              ],
                              value='y',
                              style={'margin': '15px'}
                              ),
                dbc.Label('Input points .csv file'),
                dbc.Input(
                    id='input_csv',
                    placeholder='Enter the file name..',
                    debounce=True,
                    type='text',
                    value='',
                    style={'margin': '5px'}

                ),
                dbc.Label('Input substations .csv file'),
                dbc.Input(
                    id='input_sub',
                    placeholder='Enter the file name..',
                    debounce=True,
                    type='text',
                    value='',
                    style={'margin': '5px'}
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('CRS'),
                        dbc.Input(
                            id='crs',
                            placeholder='EPSG code..',
                            debounce=True,
                            type='number',
                            value=''
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label('Resolution [meters]'),
                        dbc.Input(
                            id='resolution',
                            placeholder='1000m',
                            debounce=True,
                            type='number',
                            min=100, max=4000, step=50,
                            value='1000'
                        ),
                    ])
                ], style={'margin': '10px'}),

                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            'RUN',
                            size='lg',
                            color='warning',
                            id='create_df', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                    dbc.Col(
                        dbc.Button(
                            'NEXT',
                            size='lg',
                            color='primary',
                            id='next_step_gis', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                ], justify='around'),
            ]),
                    width={"size": 3, "offset": 0}),

            dbc.Col(html.Div([
                dbc.Spinner(size='lg', color="#4ead84", children=[
                    dbc.Tabs([
                        dbc.Tab(label='Map', label_style={"color": "#55b298"},
                                children=[
                                    dcc.Graph(
                                        id='output_gis',
                                        figure=fig,
                                    )]
                                ),
                        dbc.Tab(label='Table',
                                label_style={"color": "#55b298"},
                                children=[
                                    dash_table.DataTable(
                                        id='datatable_gis',
                                        columns=[
                                            {"name": i, "id": i} for i in
                                            gis_columns.columns
                                        ],
                                        style_table={'height': '450px'},
                                        page_count=1,
                                        page_current=0,
                                        page_size=13,
                                        page_action='custom')]
                                ),
                    ]),
                ])
            ]), width={"size": 8, "offset": 0}, align='center'),

        ], justify="around"),
    ]),

    html.Div(id='cluster_div', children=[
        dbc.Row([
            dbc.Col(html.Div(style={'textAlign': 'center'}, children=[

                html.H5(['Cluster Sensitivity'], style={'margin': '5px',
                                                        'color': "#55b298"},
                        id='cluster_sens_header'),

                dbc.Label('Limits for the MINIMUM POINTS'),
                dcc.RangeSlider(
                    id='pts',
                    allowCross=False,
                    min=10,
                    max=2000,
                    marks={10: '10', 200: '200', 400: '400', 600: '600',
                           800: '800', 1000: '1000', 1200: '1200',
                           1400: '1400,', 1600: '1600', 1800: '1800',
                           2000: '2000'},
                    step=10,
                    value=[300, 700]
                ),

                dbc.Label('Limits for the NEIGHBOURHOOD [meters]'),
                dcc.RangeSlider(
                    id='eps',
                    allowCross=False,
                    min=100,
                    max=5000,
                    marks={100: '100', 500: '500', 1000: '1000', 1500: '1500',
                           2000: '2000', 2500: '2500', 3000: '3000',
                           3500: '3500,', 4000: '4000', 4500: '4500',
                           5000: '5000'},
                    step=100,
                    value=[1200, 1700]
                ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Spans'),
                        dbc.Input(
                            debounce=True,
                            bs_size='sm',
                            id='spans',
                            placeholder='',
                            type='number',
                            min=0, max=20, step=1,
                            value='5'),
                    ], width={"size": 4, "offset": 0}, align='center'),
                    dbc.Col([
                        dbc.Button(
                            'Sensitivity',
                            # size='sm',
                            color='warning',
                            id='bt_cluster_sens', n_clicks=0, disabled=False,
                            style={'textAlign': 'center', 'margin': '10px'},
                            className='button-primary'),
                    ], width={"size": 6, "offset": 0}, align='end')
                ], justify="around"),

                html.H5(['Cluster Analysis'], style={'margin': '5px',
                                                     'color': "#55b298"}),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Final EPS'),
                        dbc.Input(
                            debounce=True,
                            bs_size='sm',
                            id='eps_final',
                            placeholder='',
                            type='number',
                            min=0, max=10000, step=50,
                            value=1500),
                    ]),
                    dbc.Col([
                        dbc.Label('Final minPTS'),
                        dbc.Input(
                            debounce=True,
                            bs_size='sm',
                            id='pts_final',
                            placeholder='..',
                            type='number',
                            min=0, max=10000, step=10,
                            value=500),
                    ])
                ]),
                html.H6(['Choose two clusters to merge'],
                        style={'textAlign': 'center',
                               'color': "#55b298",
                               'margin': '5px'}),
                dbc.Row([
                    dbc.Col([
                        # dbc.Label('Cluster to merge'),
                        dbc.Input(
                            debounce=True,
                            bs_size='sm',
                            id='c1_merge',
                            placeholder='',
                            type='number',
                            min=0, max=99, step=1,
                            value=''),
                    ], width={"size": 3, "offset": 0}, align='center'),
                    dbc.Col([
                        # dbc.Label('Cluster'),
                        dbc.Input(
                            debounce=True,
                            bs_size='sm',
                            id='c2_merge',
                            placeholder='',
                            type='number',
                            min=0, max=99, step=1,
                            value=''),
                    ], width={"size": 3, "offset": 0}, align='center'),
                    dbc.Col([
                        dbc.Button(
                            'Merge',
                            # size='sm',
                            color='warning',
                            id='bt_merge_clusters', n_clicks=0,
                            disabled=False,
                            style={'textAlign': 'center',
                                   'margin': '0px'},
                            className='button-primary'),
                    ], width={"size": 4, "offset": 0}, align='center')
                ], justify="around", style={'height': -100}),

                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            'RUN',
                            size='lg',
                            color='warning',
                            id='cluster_analysis', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                    dbc.Col(
                        dbc.Button(
                            'NEXT',
                            size='lg',
                            color='primary',
                            id='next_step_cluster', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                ], justify='around'),
            ]), width={"size": 3, "offset": 0}),

            dbc.Col(html.Div([
                dbc.Spinner(size='lg', color="#55b298", children=[
                    dbc.Tabs([
                        dbc.Tab(label='Map', label_style={"color": "#55b298"},
                                children=[
                                    dcc.Graph(
                                        id='output_cluster',
                                        figure=fig,
                                    )]
                                ),
                        dbc.Tab(label='Sensitivity',
                                label_style={"color": "#55b298"}, children=[
                                dcc.Graph(
                                    id='output_sens',
                                    figure=fig,
                                )]
                                ),
                    ]),
                ])
            ]), width={"size": 8, "offset": 0}, align='center'),

        ], justify="around"),
    ]),

    html.Div(id='grid_div', children=[
        dbc.Row([
            dbc.Col(html.Div(style={'textAlign': 'center'}, children=[
                html.H2(['Grid Routing'], style={'color': "#55b298"}),
                dbc.Checklist(id='full_ele', switch=True, inline=True,
                              options=[
                                  {'label': 'Total Electrification',
                                   'value': 'y'},
                              ],
                              value=[],
                              style={'textAlign': 'center', 'margin': '15px'},
                              ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Population Threshold'),
                        dbc.Input(
                            id='pop_thresh',
                            placeholder='',
                            debounce=True,
                            type='number',
                            min=0, max=1000, step=10,
                            value='100')
                    ]),
                    dbc.Col([
                        dbc.Label('Line Base Cost'),
                        dbc.Input(
                            id='line_bc',
                            placeholder='[€/km]',
                            debounce=True,
                            type='number',
                            value='')
                    ]),
                    dbc.Col([
                        dbc.Label('Load per Capita'),
                        dbc.Input(
                            id='pop_load',
                            placeholder='[kW]',
                            debounce=True,
                            type='number',
                            value='')
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('HV/MV Substation cost'),
                        dbc.Input(
                            id='sub_cost_HV',
                            placeholder='Enter a value [€]..',
                            debounce=True,
                            min=0, max=999999999,
                            type='number',
                            value=''
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label('MV/LV Substation cost'),
                        dbc.Input(
                            id='sub_cost_MV',
                            placeholder='Enter a value [€]..',
                            debounce=True,
                            min=0, max=999999999,
                            type='number',
                            value=''
                        ),
                    ])
                ]),

                dbc.Checklist(id='branch', switch=True, inline=True,
                              options=[
                                  {'label': 'Branch Strategy',
                                   'value': 'y'},
                              ],
                              value=[],
                              style={'margin': '15px'}
                              ),
                dbc.Row([
                    dbc.Col([
                        dbc.Label(
                            'Population Treshold (Main Branches)'),
                        dbc.Input(
                            debounce=True,
                            id='pop_thresh_lr',
                            type='number',
                            min=0, max=1000, step=10,
                            value='200'),
                    ]),
                    dbc.Col([
                        dbc.Label(
                            'Line base cost (Collaterals)'),
                        dbc.Input(
                            debounce=True,
                            id='line_bc_col',
                            placeholder='[€/km]',
                            type='number',
                            value='')
                    ]),
                ]),

                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            'RUN',
                            size='lg',
                            color='warning',
                            id='grid_routing', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                    dbc.Col(
                        dbc.Button(
                            'NEXT',
                            size='lg',
                            color='primary',
                            id='next_step_routing', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                ], justify='around'),
            ]),
                    width={"size": 3, "offset": 0}, align='center'),

            dbc.Col(html.Div([
                dbc.Spinner(size='lg', color="#55b298", children=[
                    dbc.Tabs([
                        dbc.Tab(label='Map', label_style={"color": "#55b298"},
                                children=[
                                    dcc.Graph(
                                        id='output_grid',
                                        figure=fig,
                                    )]
                                ),
                        dbc.Tab(label='Table',
                                label_style={"color": "#55b298"},
                                children=[
                                    dash_table.DataTable(
                                        id='datatable_grid',
                                        columns=[],
                                        style_header={
                                            'whiteSpace': 'normal',
                                            'height': 'auto',
                                            'width': '30px',
                                            'textAlign': 'center'},
                                        style_table={'height': '450px'},
                                        page_count=1,
                                        page_current=0,
                                        page_size=13,
                                        page_action='custom',

                                        sort_action='custom',
                                        sort_mode='single',
                                        sort_by=[])]
                                ),
                    ]),
                ])
            ]), width={"size": 8, "offset": 0}, align='center'),

        ], justify="around"),
    ]),

    html.Div(id='mg_div', children=[
        dbc.Row([
            dbc.Col(html.Div(style={'textAlign': 'center'}, children=[
                html.H2(['Microgrid Sizing'], style={'color': "#55b298"}),
                dbc.Checklist(id='import_res', switch=True, inline=True,
                              options=[
                                  {'label': 'Import RES data',
                                   'value': 'y'},
                              ],
                              value=[],
                              style={'textAlign': 'center', 'margin': '15px'},
                              ),

                dbc.Row([
                    dbc.Col([
                        dbc.Label('Cost of Electricity'),
                        dbc.Input(
                            id='coe2',
                            placeholder='[€/kWh]',
                            debounce=True,
                            min=0, max=9999,
                            type='number',
                            value=''
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label('Inflation Rate'),
                        dbc.Input(
                            id='grid_ir2',
                            placeholder='[%/year]',
                            debounce=True,
                            min=0, max=1, step=0.01,
                            type='number',
                            value='0.01'
                        ),
                    ])
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Label('Grid Lifetime'),
                        dbc.Input(
                            id='grid_lifetime2',
                            placeholder='[y]',
                            debounce=True,
                            min=1, max=100, step=1,
                            type='number',
                            value='40'
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label('Grid O&M Costs'),
                        dbc.Input(
                            debounce=True,
                            id='grid_om2',
                            placeholder='[% of total]',
                            min=0, max=1, step=0.01,
                            type='number',
                            value='0.01'
                        ),
                    ])
                ]),
                html.Div([
                    dbc.Label('Wind Turbine Model'),
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
                ], style={'margin': '15px'}),

                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            'RUN',
                            size='lg',
                            color='warning',
                            id='mg_sizing', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                    dbc.Col(
                        dbc.Button(
                            'NEXT',
                            size='lg',
                            color='primary',
                            id='next_step_mg', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                ], justify='around'),
            ]),
                    width={"size": 3, "offset": 0}, align='center'),

            dbc.Col(html.Div([
                dbc.Spinner(size='lg', color="#55b298", children=[
                    dbc.Tabs([
                        dbc.Tab(label='Map', label_style={"color": "#55b298"},
                                children=[
                                    dcc.Graph(
                                        id='output_mg',
                                        figure=fig,
                                    )]
                                ),
                        dbc.Tab(label='Microgrids',
                                label_style={"color": "#55b298"},
                                children=[
                                    dash_table.DataTable(
                                        id='datatable_mg',
                                        columns=[
                                            {"name": i, "id": i} for i in
                                            mg_columns.columns
                                        ],
                                        style_header={
                                            'whiteSpace': 'normal',
                                            'height': 'auto',
                                            'width': '30px',
                                            'textAlign': 'center'
                                        },
                                        style_table={'height': '450px'},
                                        page_count=1,
                                        page_current=0,
                                        page_size=13,
                                        page_action='custom',
                                        sort_action='custom',
                                        sort_mode='single',
                                        sort_by=[]
                                    )

                                ]
                                ),
                    ]),
                ])
            ]), width={"size": 8, "offset": 0}, align='center'),

        ], justify="around"),

    ]),

    html.Div(id='lcoe_div', children=[
        dbc.Row([
            dbc.Col(html.Div(style={'textAlign': 'center'}, children=[
                html.H2(['LCOE Analysis'], style={'color': "#55b298"}),
                dbc.Row([
                    dbc.Col([
                        dbc.Label('Cost of Electricity'),
                        dbc.Input(
                            id='coe',
                            placeholder='[€/kWh]',
                            debounce=True,
                            min=0, max=9999,
                            type='number',
                            value=''
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label('Inflation Rate'),
                        dbc.Input(
                            id='grid_ir',
                            placeholder='[%/year]',
                            debounce=True,
                            min=0, max=1, step=0.01,
                            type='number',
                            value='0.01'
                        ),
                    ])
                ]),

                dbc.Row([
                    dbc.Col([
                        dbc.Label('Grid Lifetime'),
                        dbc.Input(
                            id='grid_lifetime',
                            placeholder='[y]',
                            debounce=True,
                            min=1, max=100, step=1,
                            type='number',
                            value='40'
                        ),
                    ]),
                    dbc.Col([
                        dbc.Label('Grid O&M Costs'),
                        dbc.Input(
                            debounce=True,
                            id='grid_om',
                            placeholder='[% of total]',
                            min=0, max=1, step=0.01,
                            type='number',
                            value='0.01'
                        ),
                    ])
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Button(
                            'RUN',
                            size='lg',
                            color='warning',
                            id='lcoe_btn', n_clicks=0,
                            style={'textAlign': 'center', 'margin': '10px'},
                        ), width={"size": 6, "offset": 0}, align='center'),
                ], justify='start'),
            ]),
                    width={"size": 3, "offset": 0}, align='center'),

            dbc.Col(html.Div([
                dbc.Spinner(size='lg', color="#55b298", children=[
                    dbc.Tabs([
                        dbc.Tab(label='Map', label_style={"color": "#55b298"},
                                children=[
                                    dcc.Graph(
                                        id='output_lcoe',
                                        figure=fig,
                                    )]
                                ),
                        dbc.Tab(label='LCOE',
                                label_style={"color": "#55b298"},
                                children=[
                                    dash_table.DataTable(
                                        id='datatable_lcoe',
                                        columns=[
                                            {"name": i, "id": i} for i in
                                            lcoe_columns.columns
                                        ],
                                        style_header={
                                            'whiteSpace': 'normal',
                                            'height': 'auto',
                                            'width': '30px',
                                            'textAlign': 'center'
                                        },
                                        style_table={'height': '450px'},
                                        page_count=1,
                                        page_current=0,
                                        page_size=13,
                                        page_action='custom',
                                        sort_action='custom',
                                        sort_mode='single',
                                        sort_by=[]
                                    )

                                ]
                                ),
                    ]),
                ])
            ]), width={"size": 8, "offset": 0}, align='center'),

        ], justify="around"),

    ]),
    html.P(id='config_out', style={'display': 'none'}),
    dbc.Tooltip(
        "WARNING: This option increases the computation time, ",
        target="full_ele",
    ),
    dbc.Tooltip(
        "WARNING: This option greatly increases the computation time, ",
        target="branch",
    ),
    dbc.Tooltip(
        "Select a range of values for the input parameters and the number"
        "of spans between this interval to show in the sensitivity graph."
        "The goal should be to maximize the % of clustered people maintaining "
        "a high value of people/km²."
        ,
        target="cluster_sens_header",
    ),
])


@app.callback(Output('step', 'value'),
              [Input('next_step_start', 'n_clicks'),
               Input('next_step_gis', 'n_clicks'),
               Input('next_step_cluster', 'n_clicks'),
               Input('next_step_routing', 'n_clicks'),
               Input('next_step_mg', 'n_clicks')])
def go_next_step(next_step_start, next_step_gis, next_step_cluster,
                 next_step_routing, next_step_mg):
    """ Changes the value of the step selection according to which NEXT button
    that was pressed"""
    button_pressed = dash.callback_context.triggered[0]['prop_id'].split('.')[
        0]
    if button_pressed == 'next_step_start':
        return 1
    if button_pressed == 'next_step_gis':
        return 2
    if button_pressed == 'next_step_cluster':
        return 3
    if button_pressed == 'next_step_routing':
        return 4
    if button_pressed == 'next_step_mg':
        return 5
    else:
        return 0


@app.callback(Output('start_div', 'style'),
              [Input('step', 'value')])
def change_interface(step):
    """ Changes the html.Div of whole page according to the step selected """
    if step == 0:
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback(Output('gis_div', 'style'),
              [Input('step', 'value')])
def change_interface(step):
    """ Changes the html.Div of whole page according to the step selected """
    if step == 1:
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback(Output('cluster_div', 'style'),
              [Input('step', 'value')])
def change_interface(step):
    """ Changes the html.Div of whole page according to the step selected """
    if step == 2:
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback(Output('grid_div', 'style'),
              [Input('step', 'value')])
def change_interface(step):
    """ Changes the html.Div of whole page according to the step selected """
    if step == 3:
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback(Output('mg_div', 'style'),
              [Input('step', 'value')])
def change_interface(step):
    """ Changes the html.Div of whole page according to the step selected """
    if step == 4:
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback(Output('lcoe_div', 'style'),
              [Input('step', 'value')])
def change_interface(step):
    """ Changes the html.Div of whole page according to the step selected """
    if step == 5:
        return {'textAlign': 'center'}
    else:
        return {'display': 'none'}


@app.callback([Output('pop_thresh_lr', 'disabled'),
               Output('line_bc_col', 'disabled')],
              [Input('branch', 'value')])
def branch_options(branch):
    """ Enables or not the options for branch technique according to switch"""
    if not branch:
        return True, True
    else:
        return False, False


@app.callback([Output('c1_merge', 'disabled'),
               Output('c2_merge', 'disabled'),
               Output('bt_merge_clusters', 'disabled')],
              [Input('cluster_analysis', 'n_clicks')])
def merge_options(cluster_analysis):
    """ Enables or not the options for merging clusters after analysis"""
    if cluster_analysis == 0:
        return True, True, True
    else:
        return False, False, False


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
               Input('grid_lifetime', 'value'),
               Input('eps', 'value'),
               Input('pts', 'value'),
               Input('spans', 'value'),
               Input('eps_final', 'value'),
               Input('pts_final', 'value'),
               Input('c1_merge', 'value'),
               Input('c2_merge', 'value')])
def configuration(import_data, input_csv, input_sub, crs, resolution,
                  pop_thresh, line_bc, pop_load, sub_cost_HV, sub_cost_MV,
                  branch, pop_thresh_lr, line_bc_col, full_ele, wt, coe,
                  grid_ir, grid_om, grid_lifetime, eps, pts, spans, eps_final,
                  pts_final, c1_merge, c2_merge):
    """ Reads every change of input in the UI and updates the configuration
    file accordingly """
    ctx = dash.callback_context
    if ctx.triggered[0]['value'] is not None:
        para_index = config[config['Parameter'] ==
                            ctx.triggered[0]['prop_id'].split('.')[0]].index

        if isinstance(ctx.triggered[0]['value'], list):

            if not ctx.triggered[0]['value']:
                config.loc[para_index, 'Value'] = 'no'
            elif ctx.triggered[0]['value'][0] == 'y':
                config.loc[para_index, 'Value'] = 'yes'
            elif isinstance(ctx.triggered[0]['value'][0], int):
                config.values[para_index[0], 1] = ctx.triggered[0]['value']
        else:
            config.loc[para_index, 'Value'] = ctx.triggered[0]['value']
        msg = 'Parameter changed: ' + str(
            ctx.triggered[0]['prop_id'].split('.')[0])
        print(config)
    else:
        raise PreventUpdate
    return html.Div(msg)


@app.callback([Output('output_gis', 'figure'),
               Output('datatable_gis', 'page_count')],
              [Input('create_df', 'n_clicks')])
def create_dataframe(create_df):
    """ Runs the functions for creating the geodataframe when the button RUN
    present in the GIS interface is pressed """
    data_import = config.iloc[0, 1]
    input_csv = config.iloc[1, 1]
    crs = int(config.iloc[3, 1])
    resolution = float(config.iloc[4, 1])
    unit = 1
    step = 1
    if create_df >= 1:
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
        geo_df = geo_df.to_crs(epsg=4326)
        fig2 = go.Figure(go.Scattermapbox(

            name='GeoDataFrame',
            lat=geo_df.geometry.y,
            lon=geo_df.geometry.x,
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=10,
                color=geo_df.Weight,
                opacity=0.8
            ),
            text=geo_df.ID,
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

        return fig2, round(geo_df.shape[0] / 13, 0) + 1
    else:
        return dash.no_update, 1


@app.callback(Output('output_sens', 'figure'),
              [Input('bt_cluster_sens', 'n_clicks')])
def cluster_sensitivity(bt_cluster_sens):
    """ Checks if the button SENSITIVITY was pressed, if yes run the code and
     and changes the bt_out to a value that will call the change_interface
     function """
    if bt_cluster_sens > 0:
        resolution = float(config.iloc[4, 1])
        eps = list(config.iloc[20, 1])
        pts = list(config.iloc[21, 1])
        spans = int(config.iloc[22, 1])

        geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
        loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
        pop_points = pd.DataFrame(data=loc).values
        fig_sens = clustering.sensitivity2(resolution, pop_points, geo_df, eps,
                                           pts,
                                           int(spans))
        return fig_sens
    raise PreventUpdate


@app.callback(Output('output_cluster', 'figure'),
              [Input('cluster_analysis', 'n_clicks'),
               Input('bt_merge_clusters', 'n_clicks')])
def analysis(cluster_analysis, bt_merge_clusters):
    """ Checks if the button RUN GISELE was pressed, if yes run the code and
     and changes the bt_out to a value that will call the change_interface
     function """
    eps_final = int(config.iloc[23, 1])
    pts_final = int(config.iloc[24, 1])
    c1_merge = int(config.iloc[25, 1])
    c2_merge = int(config.iloc[26, 1])
    button_pressed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    pop_load = float(config.iloc[6, 1])
    if 'cluster_analysis' in button_pressed:
        geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
        loc = {'x': geo_df['X'], 'y': geo_df['Y'], 'z': geo_df['Elevation']}
        pop_points = pd.DataFrame(data=loc).values

        geo_df_clustered, clusters_list = \
            clustering.analysis2(pop_points, geo_df, pop_load,
                                 eps_final, pts_final)

        gdf_clustered_clean = geo_df_clustered[
            geo_df_clustered['Cluster'] != -1]

        fig_clusters = clustering.plot_clusters(gdf_clustered_clean,
                                                clusters_list)
        clusters_list.to_csv(r"Output/Clusters/clusters_list.csv",
                             index=False)
        geo_df_clustered.to_file(r"Output/Clusters/geo_df_clustered.json",
                                 driver='GeoJSON')
        return fig_clusters

    if 'bt_merge_clusters' in button_pressed:
        geo_df_clustered = \
            gpd.read_file(r"Output/Clusters/geo_df_clustered.json")
        geo_df_clustered.loc[geo_df_clustered['Cluster'] ==
                             c2_merge, 'Cluster'] = c1_merge
        gdf_clustered_clean = geo_df_clustered[
            geo_df_clustered['Cluster'] != -1]
        clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
        drop_index = \
            clusters_list.index[clusters_list['Cluster'] == c2_merge][0]
        clusters_list = clusters_list.drop(index=drop_index)

        fig_merged = clustering.plot_clusters(gdf_clustered_clean,
                                              clusters_list)

        clusters_list.to_csv(r"Output/Clusters/clusters_list.csv",
                             index=False)
        geo_df_clustered.to_file(r"Output/Clusters/geo_df_clustered.json",
                                 driver='GeoJSON')
        return fig_merged
    raise PreventUpdate


@app.callback(Output('output_grid', 'figure'),
              [Input('grid_routing', 'n_clicks')])
def routing(grid_routing):
    if grid_routing >= 1:
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

            grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                       resolution, line_bc,
                                                       branch, input_sub,
                                                       gdf_roads,
                                                       roads_segments)

            fig_grid = results.graph(geo_df_clustered, clusters_list, branch,
                                     grid_resume_opt, substations, pop_thresh,
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

            grid_resume_opt = optimization.connections(geo_df, grid_resume,
                                                       resolution, line_bc,
                                                       branch, input_sub,
                                                       gdf_roads,
                                                       roads_segments)

            fig_grid = results.graph(geo_df_clustered, clusters_list, branch,
                                     grid_resume_opt, substations, pop_thresh,
                                     full_ele)

        return fig_grid
    else:
        return fig


@app.callback(Output('output_mg', 'figure'),
              [Input('mg_sizing', 'n_clicks')])
def microgrid_size(mg_sizing):
    grid_lifetime = int(config.iloc[19, 1])
    wt = (config.iloc[15, 1])

    geo_df_clustered = \
        gpd.read_file(r"Output/Clusters/geo_df_clustered.json")
    clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
    clusters_list.index = clusters_list.Cluster.values

    if mg_sizing > 0:
        # exec(open('GISEle.py').read())
        load_profile, years, total_energy = load(clusters_list, grid_lifetime)

        mg = sizing(load_profile, clusters_list, geo_df_clustered, wt, years)
        fig_mg = []
        return fig
    return fig


@app.callback(Output('output_lcoe', 'figure'),
              [Input('lcoe_btn', 'n_clicks')])
def lcoe_computation(lcoe_btn):
    branch = config.iloc[11, 1]
    full_ele = config.iloc[14, 1]
    input_sub = config.iloc[2, 1]
    pop_thresh = float(config.iloc[7, 1])
    coe = float(config.iloc[16, 1])
    grid_om = float(config.iloc[18, 1])
    grid_ir = float(config.iloc[17, 1])
    grid_lifetime = int(config.iloc[19, 1])

    geo_df = gpd.read_file(r"Output/Datasets/geo_df_json")
    geo_df_clustered = \
        gpd.read_file(r"Output/Clusters/geo_df_clustered.json")
    clusters_list = pd.read_csv(r"Output/Clusters/clusters_list.csv")
    clusters_list.index = clusters_list.Cluster.values
    substations = pd.read_csv(r'Input/' + input_sub + '.csv')
    geometry = [Point(xy) for xy in zip(substations['X'], substations['Y'])]
    substations = gpd.GeoDataFrame(substations, geometry=geometry,
                                   crs=geo_df.crs)

    mg = pd.read_csv('Output/Microgrids/microgrids.csv')
    mg.index = mg.Cluster.values
    total_energy = pd.read_csv('Output/Microgrids/Grid_energy.csv')
    total_energy.index = total_energy.Cluster.values

    if lcoe_btn > 0:
        if branch == 'no':
            grid_resume_opt = pd.read_csv(r'Output/Grids/grid_resume_opt.csv')
            grid_resume_opt.index = grid_resume_opt.Cluster.values
            grid_resume = pd.read_csv(r'Output/Grids/grid_resume_opt.csv')
            grid_resume.index = grid_resume.Cluster.values

            fig_grid = results.graph(geo_df_clustered, clusters_list, branch,
                                     grid_resume_opt, substations, pop_thresh,
                                     full_ele)

            final_lcoe = lcoe_analysis(clusters_list, total_energy,
                                       grid_resume_opt, mg,
                                       coe,
                                       grid_ir, grid_om,
                                       grid_lifetime)
        elif branch == 'yes':
            grid_resume_opt = \
                pd.read_csv(r'Output/Branches/grid_resume_opt.csv')
            grid_resume_opt.index = grid_resume_opt.Cluster.values
            grid_resume = \
                pd.read_csv(r'Output/Branches/grid_resume.csv')
            grid_resume.index = grid_resume.Cluster.values
            fig_grid = results.graph(geo_df_clustered, clusters_list, branch,
                                     grid_resume_opt, substations, pop_thresh,
                                     full_ele)

            final_lcoe = lcoe_analysis(clusters_list, total_energy,
                                       grid_resume_opt, mg,
                                       coe,
                                       grid_ir, grid_om,
                                       grid_lifetime)
        return fig_grid
    else:
        return fig


@app.callback(Output('datatable_gis', 'data'),
              [Input('datatable_gis', "page_current"),
               Input('datatable_gis', "page_size"),
               Input('output_gis', "figure")])
def update_table(page_current, page_size, output_gis):
    if os.path.isfile(r'Output/Datasets/geo_df_json'):
        geo_df2 = pd.DataFrame(
            gpd.read_file(r"Output/Datasets/geo_df_json").drop(
                columns='geometry'))
    else:
        geo_df2 = pd.DataFrame()
    return geo_df2.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')


@app.callback([Output('datatable_grid', 'data'),
               Output('datatable_grid', 'columns')],
              [Input('datatable_grid', "page_current"),
               Input('datatable_grid', "page_size"),
               Input('datatable_grid', 'sort_by'),
               Input('output_grid', "figure"),
               Input('branch', 'value')])
def update_table(page_current, page_size, sort_by, output_grid, branches):
    branch = config.iloc[11, 1]
    geo_df2 = pd.DataFrame()

    if branch == 'no':
        if os.path.isfile(r'Output/Grids/grid_resume_opt.csv'):
            geo_df2 = pd.read_csv(r'Output/Grids/grid_resume_opt.csv')
            geo_df2 = geo_df2.round(2)

            if len(sort_by):
                geo_df2 = geo_df2.sort_values(
                    sort_by[0]['column_id'],
                    ascending=sort_by[0]['direction'] == 'asc',
                    inplace=False)

    if branch == 'yes':
        if os.path.isfile(r'Output/Branches/grid_resume_opt.csv'):
            geo_df2 = pd.read_csv(r'Output/Branches/grid_resume_opt.csv')
            geo_df2 = geo_df2.round(2)

            if len(sort_by):
                geo_df2 = geo_df2.sort_values(
                    sort_by[0]['column_id'],
                    ascending=sort_by[0]['direction'] == 'asc',
                    inplace=False)
    geo_df2 = geo_df2.dropna(axis=1, how='all')
    columns = [{"name": i, "id": i} for i in geo_df2.columns]
    return geo_df2.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records'), columns


@app.callback(Output('datatable_mg', 'data'),
              [Input('datatable_mg', "page_current"),
               Input('datatable_mg', "page_size"),
               Input('datatable_mg', 'sort_by'),
               Input('output_mg', "figure"),
               Input('coe2', "value")])
def update_table(page_current, page_size, sort_by, output_mg, coe2):
    if os.path.isfile(r'Output/Microgrids/microgrids.csv'):
        geo_df2 = pd.read_csv(r'Output/Microgrids/microgrids.csv')
        geo_df2 = geo_df2.round(2)

        if len(sort_by):
            geo_df2 = geo_df2.sort_values(
                sort_by[0]['column_id'],
                ascending=sort_by[0]['direction'] == 'asc',
                inplace=False)
    else:
        geo_df2 = pd.DataFrame()
    return geo_df2.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')


@app.callback(Output('datatable_lcoe', 'data'),
              [Input('datatable_lcoe', "page_current"),
               Input('datatable_lcoe', "page_size"),
               Input('datatable_lcoe', 'sort_by'),
               Input('output_lcoe', "figure")])
def update_table(page_current, page_size, sort_by, output_mg):
    if os.path.isfile(r'Output/Microgrids/LCOE_Analysis.csv'):
        geo_df2 = pd.read_csv(r'Output/Microgrids/LCOE_Analysis.csv')
        geo_df2 = geo_df2.round(2)

        if len(sort_by):
            geo_df2 = geo_df2.sort_values(
                sort_by[0]['column_id'],
                ascending=sort_by[0]['direction'] == 'asc',
                inplace=False)
    else:
        geo_df2 = pd.DataFrame()
    return geo_df2.iloc[
           page_current * page_size:(page_current + 1) * page_size
           ].to_dict('records')


if __name__ == "__main__":
    app.run_server(debug=True)
