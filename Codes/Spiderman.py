"""
Born in Tue May 18 19:19:38 2019

@author: Ragamuffin
"""

import geopandas as gpd
import pandas as pd
import math
import numpy
from Codes.Weight_matrix import *
from Codes.remove_cycles_weight_final import *
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from shapely.geometry import Point, LineString, box
from fiona.crs import from_epsg
import matplotlib.pyplot as plt
import networkx as nx
import time


def Spiderman(mesh, cluster_points, Proj_coords, paycheck,resolution):

    start_time = time.time()
    total_time = 0
    # remove not useful columns
    valid_fields = ['ID', 'X', 'Y', 'Population', 'Elevation', 'Weight']
    blacklist = []
    for x in mesh.columns:
        if x not in valid_fields:
            blacklist.append(x)
    # mesh.drop(blacklist, axis=1, inplace=True)
    # cluster_points.drop(blacklist, axis=1, inplace=True)
    # remove repetitions in the dataframe
    mesh.drop_duplicates('ID', keep='last', inplace=True)
    # cluster_points.drop_duplicates('ID', keep='last', inplace=True)
    # Creation of a pandas DataFrame containing only coordinates of populated points
    d = {'x': cluster_points['X'], 'y': cluster_points['Y'], 'z': cluster_points['Elevation']}
    pop_points = pd.DataFrame(data=d)
    pop_points.index = cluster_points['ID']
    print('pop_points has been created')

    # Compute 3D distance matrix, analysing the type of coordinate system of the input layer
    Distance_3D = pd.DataFrame(cdist(pop_points.values, pop_points.values, 'euclidean'), index=pop_points.index,
                               columns=pop_points.index)
    print('3D distance matrix has been created')
    # Minimum Spanning tree with Kruskal algorithm
    X = csr_matrix(Distance_3D)  # X is the array containing the values of all the possible connections between nodes
    Distmatrix = pd.DataFrame(Distance_3D, index=pop_points.index, columns=pop_points.index)  # Equal to Distance_3D
    Tcsr = minimum_spanning_tree(X)
    # Returns a minimum spanning tree of the undirected graph computed using Kruskal algorithm
    Adj_matrix = pd.DataFrame(Tcsr.todense(), columns=Distmatrix.columns, index=cluster_points['ID'])
    print('Minimum spanning tree analysis concluded')
    # Create a new shapefile, called 'grid' with one line connecting each couple of points given by the MST algorithm
    # Remember: in order to create a shp you have to start from a GeoDataFrame
    temp = Adj_matrix
    temp = temp.to_numpy()
    N = (temp > 0).sum()

    k = 0
    grid = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
    grid['id'] = pd.Series(range(1, N + 1))
    grid['x1'] = pd.Series(range(1, N + 1))
    grid['y1'] = pd.Series(range(1, N + 1))
    grid['x2'] = pd.Series(range(1, N + 1))
    grid['y2'] = pd.Series(range(1, N + 1))
    grid['ID1'] = pd.Series(range(1, N + 1))
    grid['ID2'] = pd.Series(range(1, N + 1))
    grid.loc[:, 'geometry'] = None
    grid['geometry'].astype = gpd.geoseries.GeoSeries
    for i, row in Adj_matrix.iterrows():
        for j, column in row.iteritems():
            if column > 0:
                grid.at[k, ['geometry']] = LineString(
                    [(pop_points.loc[i, ['x']], pop_points.loc[i, ['y']]), (pop_points.loc[j, ['x']],
                                                                            pop_points.loc[j, ['y']])])
                grid.at[k, ['id']] = 0
                grid.at[k, ['x1']] = pop_points.loc[i, ['x']].values
                grid.at[k, ['x2']] = pop_points.loc[j, ['x']].values
                grid.at[k, ['y1']] = pop_points.loc[i, ['y']].values
                grid.at[k, ['y2']] = pop_points.loc[j, ['y']].values
                grid.at[k, 'ID1'] = i
                grid.at[k, 'ID2'] = j
                k = k + 1
    indexNames = grid[grid['id'] > 0].index
    grid.drop(indexNames, inplace=True)
    grid.dropna(inplace=True)

    print('Look at the grid')
# Inserire un plot della prima rete
    print('Classification of lines:')

    # Calculate length of each line. NB: This is the 2D length: useful to distinguish lines connecting neighbor points from
    # the others
    grid['line_length'] = grid['geometry'].length
    # new dataframe with only long lines
    #min_length = grid['line_length'].min()
    diag_length = resolution*1.7
    #diag_length = min_length * math.sqrt(2)
    Long_lines = grid.loc[grid['line_length'] > math.ceil(diag_length)]
    print('Number of long lines: ', Long_lines.__len__())
    # Dataframe with only short lines
    Short_lines = grid.loc[grid['line_length'] <= math.ceil(diag_length)]
    print('Number of short lines: ', Short_lines.__len__())
    Short_lines = Short_lines.reset_index(drop=True)
    Short_lines['Weight'] = 2
    # Il peso della connessione sarà pari alla distanza moltiplicata per la media tra gli indici di penalità dei due estremi
    connections = []
    for indice, row in Short_lines.iterrows():
        connections.append([min(row['ID1'], row['ID2']), max(row['ID1'], row['ID2'])])
        travel = Distance_3D.loc[row['ID1'], row['ID2']]
        penalty1 = float(cluster_points[cluster_points['ID'] == row['ID1']].Weight)
        penalty2 = float(cluster_points[cluster_points['ID'] == row['ID2']].Weight)
        Short_lines.at[indice, 'Weight'] = paycheck/1000 * travel * (penalty1 + penalty2) / 2
    # Dataframe containing only connected points
    Lights_on = []
    for i in pop_points.index.values:
        if i in Short_lines.ID1.values or i in Short_lines.ID2.values:
            Lights_on.append(i)
    Lights_on = np.array(Lights_on)
    print('Points connected with short_lines: ', Lights_on.__len__())
    del grid

    # ------- Steiner tree algorithm for each of the selected lines ------- #
    print('Dijkstra analysis begins!')
    Dijkstra = mesh
    del mesh
    Dijkstra.loc[:, 'geometry'] = None
    # NUOVO METODO--------------------
    geometry = [Point(xy) for xy in zip(Dijkstra['X'], Dijkstra['Y'])]
    GeoDijkstra = gpd.GeoDataFrame(Dijkstra, geometry=geometry, crs=from_epsg(Proj_coords))
    print('Begin Dijkstra')
    if len(Long_lines) != 0:
        Long_lines.loc[:, 'xmin'] = 0
        Long_lines.loc[:, 'xmax'] = 0
        Long_lines.loc[:, 'ymin'] = 0
        Long_lines.loc[:, 'ymax'] = 0

        # Sort Long_lines in ascending way
        Long_lines = Long_lines.sort_values(by='line_length', ascending=True)
        Long_lines = Long_lines.reset_index(drop=True)
        print('Long lines have been sorted in ascending way')

        iteration = 0
        TOT_it = Long_lines.__len__()
        for i, row in Long_lines.iterrows():
            # Define boundaries for each Dijkstra analysis
            iteration = iteration + 1
            #print('Iterazione %d di %d' % (iteration, TOT_it))
            if Long_lines.loc[i, ['line_length']].values < 1000:
                Long_lines.at[i, ['xmin']] = min(Long_lines.loc[i, ['x1']].values, Long_lines.loc[i, ['x2']].values) - \
                                             Long_lines.loc[i, ['line_length']].values
                Long_lines.at[i, ['xmax']] = max(Long_lines.loc[i, ['x1']].values, Long_lines.loc[i, ['x2']].values) + \
                                             Long_lines.loc[i, ['line_length']].values
                Long_lines.at[i, ['ymin']] = min(Long_lines.loc[i, ['y1']].values, Long_lines.loc[i, ['y2']].values) - \
                                             Long_lines.loc[i, ['line_length']].values
                Long_lines.at[i, ['ymax']] = max(Long_lines.loc[i, ['y1']].values, Long_lines.loc[i, ['y2']].values) + \
                                             Long_lines.loc[i, ['line_length']].values
            elif Long_lines.loc[i, ['line_length']].values < 2000:
                Long_lines.at[i, ['xmin']] = min(Long_lines.loc[i, ['x1']].values, Long_lines.loc[i, ['x2']].values) - \
                                             Long_lines.loc[i, ['line_length']].values / 1.5
                Long_lines.at[i, ['xmax']] = max(Long_lines.loc[i, ['x1']].values, Long_lines.loc[i, ['x2']].values) + \
                                             Long_lines.loc[i, ['line_length']].values / 1.5
                Long_lines.at[i, ['ymin']] = min(Long_lines.loc[i, ['y1']].values, Long_lines.loc[i, ['y2']].values) - \
                                             Long_lines.loc[i, ['line_length']].values / 1.5
                Long_lines.at[i, ['ymax']] = max(Long_lines.loc[i, ['y1']].values, Long_lines.loc[i, ['y2']].values) + \
                                             Long_lines.loc[i, ['line_length']].values / 1.5
            else:
                Long_lines.at[i, ['xmin']] = min(Long_lines.loc[i, ['x1']].values, Long_lines.loc[i, ['x2']].values) - \
                                             Long_lines.loc[i, ['line_length']].values / 3
                Long_lines.at[i, ['xmax']] = max(Long_lines.loc[i, ['x1']].values, Long_lines.loc[i, ['x2']].values) + \
                                             Long_lines.loc[i, ['line_length']].values / 3
                Long_lines.at[i, ['ymin']] = min(Long_lines.loc[i, ['y1']].values, Long_lines.loc[i, ['y2']].values) - \
                                             Long_lines.loc[i, ['line_length']].values / 3
                Long_lines.at[i, ['ymax']] = max(Long_lines.loc[i, ['y1']].values, Long_lines.loc[i, ['y2']].values) + \
                                             Long_lines.loc[i, ['line_length']].values / 3
            Long_lines.at[i, ['geometry']] = box(minx=Long_lines.loc[i, ['xmin']].values,
                                                 miny=Long_lines.loc[i, ['ymin']].values,
                                                 maxx=Long_lines.loc[i, ['xmax']].values,
                                                 maxy=Long_lines.loc[i, ['ymax']].values)
            print('Checkpoint 1')
            # Filter original matrix df_in with the defined boundary: check which points are in the Polygon
            # This operation can be done only with GeoDataFrame. That's why I needed to convert Dijkstra
            df_box = GeoDijkstra[GeoDijkstra.within(Long_lines.loc[i, 'geometry'])]
            df_box.index = pd.Series(range(0, len(df_box['ID'])))
            # In order to launch Dijkstra analysis we need build the matrix of edges. Same procedure made for pop_points
            solocoord2 = {'x': df_box['X'], 'y': df_box['Y']}
            solocoord3 = {'x': df_box['X'], 'y': df_box['Y'], 'z': df_box['Elevation']}
            Battlefield2 = pd.DataFrame(data=solocoord2)
            Battlefield2.index = df_box['ID']
            Battlefield3 = pd.DataFrame(data=solocoord3)
            Battlefield3.index = df_box['ID']
            print('Checkpoint 2')

            Distance_2D_box = distance_matrix(Battlefield2, Battlefield2)
            Distance_3D_box = pd.DataFrame(cdist(Battlefield3.values, Battlefield3.values, 'euclidean'),
                                           index=Battlefield3.index, columns=Battlefield3.index)

            Weight_matrix = weight_matrix(df_box, Distance_3D_box, paycheck)
            print('Checkpoint 3')
            # Definition of weighted connections matrix
            Edges_matrix = Weight_matrix
            Edges_matrix[Distance_2D_box > math.ceil(diag_length)] = 0

            # If a specific connection already belongs to the graph, we won't incur in additional costs if we travel along it
            # Thus, we set the weight of all the already traced line to 0
            print('Checkpoint 4: Edges matrix completed')
            #print('Following connections are already part of the grid:')
            for p, rope in Short_lines.iterrows():
                if rope['ID1'] in Edges_matrix.index.values and rope['ID2'] in Edges_matrix.index.values:
                    #print(rope['ID1'], rope['ID2'])
                    Edges_matrix.loc[rope['ID1'], rope['ID2']] = 0.01
                    Edges_matrix.loc[rope['ID2'], rope['ID1']] = 0.01
            Graph = nx.from_numpy_matrix(Edges_matrix.to_numpy())
#            Graph.remove_edges_from(np.where(Edges_matrix.to_numpy() == 999999999))
            # Extrapolate from the box the terminal nodes as the two nodes of the long line
            source = df_box.loc[df_box['ID'].values == row['ID1'], :]
            target = df_box.loc[df_box['ID'].values == row['ID2'], :]
            source = source.index[0]
            target = target.index[0]
            print('Checkpoint 5')
            # Lancio Dijkstra e ottengo il percorso ottimale per connettere 'source' a 'target'
            path = nx.dijkstra_path(Graph, source, target, weight='weight')
            Steps = len(path)
            # Analisi del path
            r = 0  # Giusto un contatore
            PathID = []
            Digievoluzione = gpd.GeoDataFrame(crs=from_epsg(Proj_coords))
            Digievoluzione['id'] = pd.Series(range(1, Steps))
            Digievoluzione['x1'] = pd.Series(range(1, Steps))
            Digievoluzione['x2'] = pd.Series(range(1, Steps))
            Digievoluzione['y1'] = pd.Series(range(1, Steps))
            Digievoluzione['y2'] = pd.Series(range(1, Steps))
            Digievoluzione['ID1'] = pd.Series(range(1, Steps))
            Digievoluzione['ID2'] = pd.Series(range(1, Steps))
            Digievoluzione['Weight'] = pd.Series(range(1, Steps))
            Digievoluzione.loc[:, 'geometry'] = None
            Digievoluzione['geometry'].stype = gpd.geoseries.GeoSeries
            print('Checkpoint 6:')

            for h in range(0, Steps - 1):
                con = [min(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']),
                       max(df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID'])]
                if con not in connections:
                    PathID.append(df_box.loc[path[h], 'ID'])
                    PathID.append(df_box.loc[path[h + 1], 'ID'])
                    Digievoluzione.at[r, 'geometry'] = LineString(
                        [(df_box.loc[path[h], 'X'], df_box.loc[path[h], 'Y'], df_box.loc[path[h], 'Elevation']),
                         (
                          df_box.loc[path[h + 1], 'X'], df_box.loc[path[h + 1], 'Y'],
                          df_box.loc[path[h + 1], 'Elevation'])])
                    Digievoluzione.at[r, ['id']] = 0
                    Digievoluzione.at[r, ['x1']] = df_box.loc[path[h], 'X']
                    Digievoluzione.at[r, ['x2']] = df_box.loc[path[h + 1], 'X']
                    Digievoluzione.at[r, ['y1']] = df_box.loc[path[h], ['Y']]
                    Digievoluzione.at[r, ['y2']] = df_box.loc[path[h + 1], ['Y']]
                    Digievoluzione.at[r, 'ID1'] = df_box.loc[path[h], 'ID']
                    Digievoluzione.at[r, 'ID2'] = df_box.loc[path[h + 1], 'ID']
                    Digievoluzione.at[r, 'Weight'] = Edges_matrix.loc[
                        df_box.loc[path[h], 'ID'], df_box.loc[path[h + 1], 'ID']]
                    connections.append(con)
                    r += 1
            print('Checkpoint 7: Genomica di digievoluzione identificata')
            # Elimino le righe inutili di Digievoluzione e i doppioni da PathID
            indexNames = Digievoluzione[Digievoluzione['id'] > 0].index
            Digievoluzione.drop(indexNames, inplace=True)
            PathID = list(dict.fromkeys(PathID))
            print('Checkpoint 8: Digieevoluzione in progresso...')
            # Aggiungo i segmenti contenuti in Digievoluzione a Short_lines
            Short_lines = pd.concat([Short_lines, Digievoluzione], sort=True)
            Short_lines = Short_lines.reset_index(drop=True)
            print('Checkpoint 9: Digievoluzione completata')
            # Aggiungo i checkpoint attraversati alla lista Lights_on
            for c in PathID:
                if c not in Lights_on:
                    #print('Added nodes to Lights_on:', c)
                    Lights_on = np.append(Lights_on, c)
            #print('Lights_on length', Lights_on.__len__())
            del Digievoluzione
    print('I conti sono finiti')
    # Adesso GeoDataFrame della rete è completo e rappresentato sotto il nome di Short_lines
    Final_grid = Short_lines
    print("Cleaning useless variables")
    #del Adj_matrix, Battlefield2, Battlefield3, Distance_2D_box, Distance_3D, Distance_3D_box, Distmatrix, Edges_matrix, \
        #GeoDijkstra, Lights_on, Long_lines, PathID, Weight_matrix, connections, df_box
    Final_grid = Final_grid.drop(['id', 'x1', 'x2', 'y1', 'y2'], axis=1)
    Final_grid['line_length'] = Final_grid['geometry'].length
    cluster_points = cluster_points.drop(['X', 'Y', 'Population', 'Elevation', 'Weight'], axis=1)
    Dijkstra = Dijkstra.drop(['Population', 'geometry'], axis=1)

    print('Begin Cycle removal')
    grid_without_cycles = remove_cycles(Final_grid, cluster_points, Dijkstra)
    while len(Final_grid['ID1']) != len(grid_without_cycles['ID1']):
        Final_grid = grid_without_cycles
        grid_without_cycles = remove_cycles(Final_grid, cluster_points, Dijkstra)
    print('end cycle evaluation')
    total_cost = Final_grid['Weight'].sum(axis=0)
    total_length = Final_grid['line_length'].sum(axis=0)

    final_time = time.time()
    total_time = final_time - start_time
    print("Total time for tree's evolution algorithm was: " + str(total_time) + "seconds")

    return Final_grid, total_cost, total_length, connections







