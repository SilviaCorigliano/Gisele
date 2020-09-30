import os
import pandas as pd
from matplotlib.pyplot import plot
import geopandas as gpd
import plotly.graph_objs as go
from plotly.offline import plot

import dash
import dash_core_components as dcc
import dash_html_components as html


def start_app(df, clusters_list, branch, grid_resume_opt, substations):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    colors = {'background': '#111111', 'text': '#000000'}

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    fig = graph(df, clusters_list, branch, grid_resume_opt, substations)
    mg = pd.read_csv(r'Output/Microgrids/microgrids.csv')
    lcoe = pd.read_csv(r'Output/Microgrids/LCOE_Analysis.csv')
    app.layout = html.Div(children=[
        html.H1(
            children='GISEle: GIS for Electrification',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),

        html.Div(children='Developed by Politecnico di Milano', style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        html.Div([
            dcc.Tabs([
                dcc.Tab(label='Topological View', children=[
                    dcc.Graph(
                        id='example-graph-2',
                        figure=fig
                             )
                        ]),

                dcc.Tab(label='Grid Planning Resume',
                        children=[html.Div(children=[
                            html.H4(children='Grid Planning Resume'),
                            generate_table(grid_resume_opt)])

                        ]),
                dcc.Tab(label='Microgrids',
                        children=[html.Div(children=[
                            html.H4(children='Grid Planning Resume'),
                            generate_table(mg)])

                        ]),

                dcc.Tab(label='LCOE Analysis',
                        children=[html.Div(children=[
                            html.H4(children='Grid Planning Resume'),
                            generate_table(lcoe)])

                        ]),
                 ])
        ]),
    ])

    app.run_server(debug=False)


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


def graph(df, clusters_list, branch, grid_resume_opt, substations, pop_thresh,
          full_ele='no'):
    print('Plotting results..')

    study_area = gpd.read_file(r'Input/Namanjavira_4326.geojson')
    # study_area = gpd.read_file(r'Input/Cavalcante_4326.geojson')
    study_area = study_area.to_crs(epsg=4326)

    if branch == 'yes':
        os.chdir(r'Output//Branches')
    else:
        os.chdir(r'Output//Grids')

    substations = substations.to_crs(epsg=4326)
    df = df.to_crs(epsg=4326)
    clusters = df[df['Cluster'] != -1]
    area = study_area.boundary.unary_union.xy
    if full_ele == 'no':
        terminal_nodes = clusters[clusters['Population'] >= pop_thresh]
    elif full_ele == 'yes':
        terminal_nodes = df[df['Population'] >= pop_thresh]

    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(name='Study Area',
                                   lat=list(area[1]),
                                   lon=list(area[0]),
                                   mode='lines',
                                   marker=go.scattermapbox.Marker(
                                       size=10,
                                       color='black',
                                       opacity=1
                                   ),
                                   below="''",
                                   ))

    fig.add_trace(go.Scattermapbox(name='Clusters',
                                   lat=clusters.geometry.y,
                                   lon=clusters.geometry.x,
                                   mode='markers',
                                   marker=go.scattermapbox.Marker(
                                       size=10,
                                       color='gray',
                                       opacity=0.7
                                   ),
                                   text=clusters.Cluster,
                                   hoverinfo='text',
                                   below="''"
                                   ))

    fig.add_trace(go.Scattermapbox(name='Substations',
                                   lat=substations.geometry.y,
                                   lon=substations.geometry.x,
                                   mode='markers',
                                   marker=go.scattermapbox.Marker(
                                       size=12,
                                       color='black',
                                       opacity=0.8,
                                   ),
                                   text=substations.Type,
                                   hoverinfo='text',
                                   below="''"
                                   ))

    fig.add_trace(go.Scattermapbox(name='Terminal Nodes',
                                   lat=terminal_nodes.geometry.y,
                                   lon=terminal_nodes.geometry.x,
                                   mode='markers',
                                   marker=go.scattermapbox.Marker(
                                       size=6,
                                       color='yellow',
                                       opacity=1
                                   ),
                                   text=terminal_nodes.ID,
                                   hoverinfo='text',
                                   below="''"
                                   ))

    for cluster_n in clusters_list.Cluster:

        if branch == 'yes':
            line_break('Branch_', cluster_n, fig, 'red')
            if grid_resume_opt.loc[cluster_n, 'Branch Length [km]'] != 0:
                line_break('Collateral_', cluster_n, fig, 'black')

        else:
            line_break('Grid_', cluster_n, fig, 'black')

        if grid_resume_opt.loc[cluster_n, 'Connection Length [km]'] == 0:
            continue
        line_break('Connection_', cluster_n, fig, 'blue')

    if full_ele == 'yes':
        line_break(r'all_link/', 'all_link', fig, 'green')

    fig.update_layout(mapbox_style="carto-positron",
                      mapbox_zoom=8.5)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox_center={"lat": df.geometry.y[0],
                                     "lon": df.geometry.x[0]})
    fig.update_layout(clickmode='event+select')
    fig.update_layout(showlegend=True)

    # plot(fig)
    print('Results successfully plotted')
    os.chdir(r'..//..')

    return fig


def line_break(file, cluster_n, fig, color):
    lines = gpd.read_file(file + str(cluster_n) + '.shp')
    lines = lines.to_crs(epsg=4326)
    coordinates = pd.DataFrame(columns=['x', 'y'], dtype='int64')
    count = 0
    for i in lines.iterrows():
        coordinates.loc[count, 'x'] = i[1].geometry.xy[0][0]
        coordinates.loc[count, 'y'] = i[1].geometry.xy[1][0]
        count += 1
        coordinates.loc[count, 'x'] = i[1].geometry.xy[0][1]
        coordinates.loc[count, 'y'] = i[1].geometry.xy[1][1]
        coordinates = coordinates.append(pd.Series(), ignore_index=True)
        count += 2

    fig.add_trace(go.Scattermapbox(name=file + str(cluster_n),
                                   lat=list(coordinates.y),
                                   lon=list(coordinates.x),
                                   mode='lines',
                                   marker=go.scattermapbox.Marker(
                                       size=5,
                                       color=color,
                                       opacity=1
                                   ),
                                   text=file + str(cluster_n),
                                   hoverinfo='text',
                                   below="''"
                                   ))

    return
