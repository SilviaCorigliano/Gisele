import os
import numpy as np
# from fiona.crs import from_epsg
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from matplotlib.pyplot import plot, ion, show
import geopandas as gpd
import plotly.graph_objs as go
from plotly.offline import plot


def graph():

    are_clusters_ok = 'n'
    os.chdir(r'Output//Clusters')

    lines = gpd.read_file('Grid_0.shp')
    lines = lines.to_crs(epsg=4326)
    nodes = list(zip(lines.ID1.astype(int), lines.ID2.astype(int)))

    geodf = gpd.read_file('geo_df_clustered_clean.shp')
    geodf = geodf.to_crs(epsg=4326)

    clust = gpd.read_file('geo_df_clustered.shp')
    clust = clust.to_crs(epsg=4326)
    area = clust.unary_union.convex_hull.boundary.xy

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
                                   lat=geodf.geometry.y,
                                   lon=geodf.geometry.x,
                                   mode='markers',
                                   marker=go.scattermapbox.Marker(
                                       size=10,
                                       color=geodf.Cluster,
                                       opacity=0.9
                                   ),
                                   text=geodf.Cluster,
                                   hoverinfo='text',
                                   below="''"
                                   ))

    for i in nodes:
        fig.add_trace(go.Scattermapbox(
            lat=[geodf[geodf['ID'] == i[0]].geometry.y.values[0],
                 geodf[geodf['ID'] == i[1]].geometry.y.values[0]],
            lon=[geodf[geodf['ID'] == i[0]].geometry.x.values[0],
                 geodf[geodf['ID'] == i[1]].geometry.x.values[0]],
            mode='markers+lines',
            marker=go.scattermapbox.Marker(
                size=10,
                color='rgb(0, 0, 0)',
                opacity=0.9
            ),
            text='Line',
            hoverinfo='text',
            below="''",
            legendgroup='grid',
            showlegend=False
        ))

    fig.add_trace(go.Scattermapbox(name='grid',
                                   lat=[geodf[geodf['ID'] == i[
                                       0]].geometry.y.values[0], geodf[
                                            geodf['ID'] == i[
                                                1]].geometry.y.values[0]],
                                   lon=[geodf[geodf['ID'] == i[
                                       0]].geometry.x.values[0], geodf[
                                            geodf['ID'] == i[
                                                1]].geometry.x.values[0]],
                                   mode='markers+lines',
                                   marker=go.scattermapbox.Marker(
                                       size=10,
                                       color='rgb(0, 0, 0)',
                                       opacity=0.9
                                   ),
                                   text='Line',
                                   hoverinfo='text',
                                   below="''",
                                   legendgroup='grid',
                                   ))

    fig.update_layout(mapbox_style="carto-positron",
                      mapbox_zoom=8.5)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox_center={"lat": -13.8, "lon": -47.4})
    fig.update_layout(clickmode='event+select')
    fig.update_layout(showlegend=True)
    # fig.update_layout(legend={'traceorder':'grouped'})
    plot(fig)
    print('continue computation')
    check = str(input('Would you like to merge clusters? (y/n): '))


