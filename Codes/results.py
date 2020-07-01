import os

from matplotlib.pyplot import plot
import geopandas as gpd
import plotly.graph_objs as go
from plotly.offline import plot


def graph(df, clusters_list, step, grid_resume_opt, substations):
    print('Plotting results.....')
    # os.chdir(r'Output//Clusters')

    os.chdir(r'Output//Grids')
    if step == 4:
        os.chdir(r'Output//Main Branch')
    substations = substations.to_crs(epsg=4326)
    df = df.to_crs(epsg=4326)
    clusters = df[df['Cluster'] != -1]
    area = df.unary_union.convex_hull.boundary.xy

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
                                       size=8,
                                       color=clusters.Cluster,
                                       opacity=0.9
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
                                       opacity=1,
                                   ),
                                   text=substations.Type,
                                   hoverinfo='text',
                                   below="''"
                                   ))

    for cluster_n in clusters_list.Cluster:
        grid = gpd.read_file('Grid_' + str(cluster_n) + '.shp')
        grid = grid.to_crs(epsg=4326)
        grid_nodes = list(zip(grid.ID1.astype(int), grid.ID2.astype(int)))

        for i in grid_nodes:
            fig.add_trace(go.Scattermapbox(
                lat=[df[df['ID'] == i[0]].geometry.y.values[0],
                     df[df['ID'] == i[1]].geometry.y.values[0]],
                lon=[df[df['ID'] == i[0]].geometry.x.values[0],
                     df[df['ID'] == i[1]].geometry.x.values[0]],
                mode='lines',
                marker=go.scattermapbox.Marker(
                    size=5,
                    color='rgb(0, 0, 0)',
                    opacity=0.9
                ),
                text='Grid ' + str(cluster_n),
                hoverinfo='text',
                below="''",
                legendgroup='Grid ' + str(cluster_n),
                showlegend=False
            ))

        fig.add_trace(go.Scattermapbox(name='Grid ' + str(cluster_n),
                                       lat=[df[df['ID'] == i[
                                           0]].geometry.y.values[0], df[
                                                df['ID'] == i[
                                                    1]].geometry.y.values[0]],
                                       lon=[df[df['ID'] == i[
                                           0]].geometry.x.values[0], df[
                                                df['ID'] == i[
                                                    1]].geometry.x.values[0]],
                                       mode='lines',
                                       marker=go.scattermapbox.Marker(
                                           size=5,
                                           color='rgb(0, 0, 0)',
                                           opacity=0.9
                                       ),
                                       text='Grid ' + str(cluster_n),
                                       hoverinfo='text',
                                       below="''",
                                       legendgroup='Grid_' + str(cluster_n),
                                       ))

        if grid_resume_opt.loc[cluster_n, 'Connection Length'] == 0:
            continue

        connection = gpd.read_file('Connection_' + str(cluster_n) + '.shp')
        connection = connection.to_crs(epsg=4326)
        connection_nodes = list(
            zip(connection.ID1.astype(int), connection.ID2.astype(int)))

        for i in connection_nodes:
            fig.add_trace(go.Scattermapbox(
                lat=[df[df['ID'] == i[0]].geometry.y.values[0],
                     df[df['ID'] == i[1]].geometry.y.values[0]],
                lon=[df[df['ID'] == i[0]].geometry.x.values[0],
                     df[df['ID'] == i[1]].geometry.x.values[0]],
                mode='lines',
                marker=go.scattermapbox.Marker(
                    size=5,
                    color='blue',
                    opacity=0.9
                ),
                text='Connection ' + str(cluster_n),
                hoverinfo='text',
                below="''",
                legendgroup='Connection ' + str(cluster_n),
                showlegend=False
            ))

        fig.add_trace(go.Scattermapbox(name='Connection ' + str(cluster_n),
                                       lat=[df[df['ID'] == i[
                                           0]].geometry.y.values[0], df[
                                                df['ID'] == i[
                                                    1]].geometry.y.values[0]],
                                       lon=[df[df['ID'] == i[
                                           0]].geometry.x.values[0], df[
                                                df['ID'] == i[
                                                    1]].geometry.x.values[0]],
                                       mode='lines',
                                       marker=go.scattermapbox.Marker(
                                           size=5,
                                           color='blue',
                                           opacity=0.9
                                       ),
                                       text='Connection ' + str(cluster_n),
                                       hoverinfo='text',
                                       below="''",
                                       legendgroup='Connection '
                                                   + str(cluster_n),
                                       ))

    fig.update_layout(mapbox_style="carto-positron",
                      mapbox_zoom=8.5)
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      mapbox_center={"lat": -17.19, "lon": 36.45})
    fig.update_layout(clickmode='event+select')
    fig.update_layout(showlegend=True)
    plot(fig)

    return
