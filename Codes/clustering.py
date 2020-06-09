import os
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import plotly.graph_objs as go
from plotly.offline import plot
from supporting_GISEle2 import l, s


def sensitivity(pop_points, geo_df):
    s()
    print("2.Clustering - Sensitivity Analysis")
    s()
    print(
        "In order to define which areas are sufficiently big and enough "
        "populated to be considered valuable for electrification,\ntwo "
        "parameters must be set: NEIGHBOURHOOD and MINIMUM POINTS.\n"
        "First, provide a RANGE for these parameters and also the number of "
        "SPANS between the range values.")

    check = "n"
    while check == "y":
        s()
        eps = input("Provide the limits for the NEIGHBOURHOOD parameter \n"
                    "(Example 250, 2500): ")
        eps = sorted(list(map(int, eps.split(','))))
        s()
        pts = input("Provide the limits for the MINIMUM POINTS parameter:\n"
                    "(Example 10, 100):  ")
        pts = sorted(list(map(int, pts.split(','))))
        s()
        spans = int(input("How many SPANS between these values?: "))
        s()
        span_eps = []
        span_pts = []
        for j in range(1, spans + 1):
            if spans != 1:
                eps_ = int(eps[0] + (eps[1] - eps[0]) / (spans - 1) * (j - 1))
            else:
                eps_ = int(eps[0] + (eps[1] - eps[0]) / spans * (j - 1))
            span_eps.append(eps_)
        for i in range(1, spans + 1):
            if spans != 1:
                pts_ = int(pts[0] + (pts[1] - pts[0]) / (spans - 1) * (i - 1))
            else:
                pts_ = int(pts[0] + (pts[1] - pts[0]) / spans * (i - 1))
            span_pts.append(pts_)
        tab_area = tab_people = tab_people_area = tab_cluster = \
            pd.DataFrame(index=span_eps, columns=span_pts)
        total_people = int(geo_df['Population'].sum(axis=0))
        for eps in span_eps:
            for pts in span_pts:
                db = DBSCAN(eps=eps, min_samples=pts, metric='euclidean'). \
                    fit(pop_points, sample_weight=geo_df['Population'])
                labels = db.labels_
                n_clusters_ = int(len(set(labels)) -
                                  (1 if -1 in labels else 0))
                geo_df['clusters'] = labels
                gdf_pop_noise = geo_df[geo_df['clusters'] == -1]
                noise_people = round(sum(gdf_pop_noise['Population']), 0)
                clustered_area = len(geo_df) - len(gdf_pop_noise)
                perc_area = int(clustered_area / len(geo_df) * 100)
                clustered_people = int((1 - noise_people / total_people) * 100)
                if clustered_area != 0:  # check to avoid crash by divide zero
                    people_area = (total_people - noise_people) \
                                  / clustered_area
                else:
                    people_area = 0
                tab_cluster.at[eps, pts] = n_clusters_
                tab_people.at[eps, pts] = clustered_people
                tab_area.at[eps, pts] = perc_area
                tab_people_area.at[eps, pts] = people_area
        print(
            "Number of clusters - columns MINIMUM POINTS - rows NEIGHBOURHOOD")
        print(tab_cluster)
        l()
        print(
            "% of clustered people - columns MINIMUM POINTS - rows "
            "NEIGHBOURHOOD")
        print(tab_people)
        l()
        print(
            "% of clustered area - columns MINIMUM POINTS - rows "
            "NEIGHBOURHOOD")
        print(tab_area)
        l()
        print("People per area - columns MINIMUM POINTS - rows NEIGHBOURHOOD")
        print(tab_people_area)
        l()
        check = str(input("Would you like to run another sensitivity "
                          "analysis with a more precise interval? (y/n): "))
        s()
    # os.chdir(r'Output//Clusters//Sensitivity')
    # tab_cluster.to_csv("n_clusters.csv")
    # tab_people.to_csv("%_peop.csv")
    # tab_area.to_csv("%_area.csv")
    # tab_people_area.to_csv("people_area.csv")
    # os.chdir(r'..//..//..')
    print("Clustering sensitivity process completed.\n"
          "You can check the tables in the Output folder.")
    l()

    return


def analysis(pop_points, geo_df, pop_load):
    print('Choose the final combination of NEIGHBOURHOOD and MINIMUM POINTS')
    l()
    # eps = float(input("Chosen NEIGHBOURHOOD: "))
    eps = 1500
    # eps = 3100
    # pts = 20
    s()
    pts = 500
    # pts = int(input("Chosen MINIMUM POINTS: "))
    s()
    db = DBSCAN(eps=eps, min_samples=pts, metric='euclidean').fit(
        pop_points, sample_weight=geo_df['Population'])
    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # ignore noise
    clusters_list = np.arange(n_clusters)
    print('Initial number of clusters: %d' % n_clusters)
    # print('Estimated number of noise points: %d' % list(labels).count(-1))

    check = "y"
    while check == "y":
        geo_df_clustered = geo_df
        geo_df_clustered['clusters'] = labels
        gdf_clustered_clean = geo_df_clustered[
            geo_df_clustered['clusters'] != -1]

        fig = go.Figure()
        for cluster_n in clusters_list:
            plot_cluster = gdf_clustered_clean[gdf_clustered_clean
                                               ['clusters'] == cluster_n]
            plot_cluster = plot_cluster.to_crs(epsg=4326)
            plot_cluster.X = plot_cluster.geometry.x
            plot_cluster.Y = plot_cluster.geometry.y
            fig.add_trace(go.Scattermapbox(
                lat=plot_cluster.Y,
                lon=plot_cluster.X,
                mode='markers',
                name='Cluster ' + str(cluster_n),
                marker=go.scattermapbox.Marker(
                    size=10,
                    color=cluster_n,
                    opacity=0.9
                ),
                text=plot_cluster.clusters,
                hoverinfo='text',
                below="''"
            ))
            fig.update_layout(mapbox_style="carto-positron",
                              mapbox_zoom=8.5)
            fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                              mapbox_center={"lat": plot_cluster.Y.values[0],
                                             "lon": plot_cluster.X.values[0]})

            fig.update_layout(clickmode='event+select')

        plot(fig)
        check = str(input('Would you like to merge clusters? (y/n): '))
        s()
        if check == "n":
            break
        print('The current clusters available for merging are:'
              '\n' + str(clusters_list))
        s()
        merge1 = int(input('Select a base cluster that you want to merge: '))
        merge2 = input('Which clusters would you like to merge into the '
                       'cluster' + str(merge1) + '? (Example: 1,13,25): ')
        merge2 = sorted(list(map(int, merge2.split(','))))
        s()
        for i in merge2:
            clusters_list = np.delete(clusters_list,
                                      np.where(clusters_list == i))
            labels[labels == i] = merge1

    clusters_list = np.tile(clusters_list, (3, 1))
    for i in clusters_list[0]:
        pop_i = sum(geo_df_clustered.loc
                    [geo_df_clustered['clusters'] == i, 'Population'])
        clusters_list[1, np.where(clusters_list[0] == i)] = pop_i
        clusters_list[2, np.where(clusters_list[0] == i)] = pop_i * pop_load

    os.chdir(r'Output//Clusters')
    geo_df_clustered.to_file("geo_df_clustered.shp")
    gdf_clustered_clean.to_file("geo_df_clustered_clean.shp")
    os.chdir(r'..//..')

    print("Clustering completed and files exported")
    return geo_df_clustered, clusters_list
