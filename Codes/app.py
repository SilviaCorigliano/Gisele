import os
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from Codes import initialization

if not os.path.isdir(r'Input'):
    os.chdir('..')

fig = go.Figure(go.Scattermapbox(
    lat=[''],
    lon=[''],
    mode='markers'))

fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([

    html.Div([html.H1('GISEle: GIS for Electrification',
                      style={'textAlign': 'center', 'color': '000000'}
                      ),

              html.P('Developed by Politecnico di Milano', style={
                  'textAlign': 'center',
                  'color': '000000'})
              ]),

    html.Div([
        html.Div([
            html.Label('Step Selection'),
            dcc.Slider(id='step',
                       min=0,
                       max=5,
                       marks={0: 'Start',
                              1: 'GeoDataFrame', 2: 'Clustering',
                              3: 'Grid Routing', 4: 'Microgrid Sizing',
                              5: 'LCOE Analysis'},
                       value=5,
                       ),
            dcc.Tabs([
                dcc.Tab(label='GIS Configuration', children=[
                    html.Label('Import GIS Data'),
                    dcc.RadioItems(id='import_gis',
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
                        type='text',
                        value=''
                    ),
                    html.Label('Input substations .csv file'),
                    dcc.Input(
                        id='input_sub',
                        placeholder='Enter the file name..',
                        type='text',
                        value=''
                    ),
                    html.Label('Coordinate Reference System'),
                    dcc.Input(
                        id='crs',
                        placeholder='Enter the EPSG code..',
                        type='number',
                        value=''
                    ),
                    html.Label('Resolution'),
                    dcc.Input(
                        id='resolution',
                        placeholder='1000m',
                        type='number',
                        min=100, max=4000, step=50,
                        value='1000'
                    ),
                ]),
                dcc.Tab(label='Electrical Configuration', children=[

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
                                        'value': 'branch'},
                                       {'label': 'Steiner',
                                        'value': 'steiner'},
                                   ],
                                   value='steiner',
                                   labelStyle={'display': 'inline-block'}
                                   ),
                    html.Div(id='extra_options'),

                    dcc.Checklist(
                        options=[
                            {'label': 'Total Electrification',
                             'value': 'full_ele'},
                        ],
                        value=['']
                    )
                ]),
                dcc.Tab(label='Microgrid Parameters', children=[
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
                        options=[
                            {'label': 'Nordex N27 150', 'value': 'NOR27'},
                            {'label': 'Alstom Eco 80', 'value': 'ALS80'},
                            {'label': 'Enercon E40 500', 'value': 'ENE40'},
                            {'label': 'Vestas V27 225', 'value': 'VES27'}
                        ],
                        value='NOR27'
                    )
                ]),
            ]),
            html.P(id='bt_out'),

            html.Div([
                html.Div([html.Button('Run GISEle',
                                      id='bt1', n_clicks=0)], style={
                    'textAlign': 'center'})
            ]),

            html.Div([
                html.Div([html.Button('Show Results',
                                      id='bt2', n_clicks=0)], style={
                    'textAlign': 'center'})
            ])
        ], style={'textAlign': 'center'}, className="four columns"),
        html.Div([dcc.Graph(
            id='example-graph',
            figure=fig,
            className="eight columns")])
    ], className='row')
])


@app.callback(Output('extra_options', 'children'),
              [Input('branch', 'value')])
def extra_options(branch):
    if branch == 'branch':
        opts = [
            html.Label('Population Treshold (Main Branches)'),
            dcc.Input(
                id='pop_thresh_lr',
                placeholder='Enter a value [kW]..',
                type='number',
                min=0, max=1000, step=10,
                value='200'),

            html.Label('Line base cost (Collaterals)'),
            dcc.Input(
                id='line_bc_col',
                placeholder='Enter a value [kW]..',
                type='number',
                value=''),
        ]
    else:
        opts = ''
    return opts


@app.callback(Output('bt_out', 'children'),
              [Input('bt1', 'n_clicks'),
               Input('bt2', 'n_clicks'),
               Input('step', 'value')])
def pressed_button(bt1, bt2, step):
    button_pressed = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'bt1' in button_pressed:
        clusters_list, grid_resume, mg, total_energy, coe, grid_ir, grid_om, \
        grid_lifetime = initialization.import_csv_file(step)
        print(clusters_list)
        msg = ''
    elif 'bt2' in button_pressed:
        msg = ''
    else:
        msg = ''
    return html.Div(msg)


app.run_server(debug=True)
