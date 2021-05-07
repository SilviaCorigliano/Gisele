import pandas as pd
from shapely.geometry import Point, box, LineString, MultiPoint,MultiLineString
import geopandas as gpd
import os
import numpy as np
def process(gisele_folder,case_study):
    read_input = gisele_folder+'\Case studies/'+ case_study+ '\Input\MILP\MILP_input/'
    read_output=gisele_folder+'\Case studies/' + case_study+'/Input\MILP\MILP_output/'
    read_general_data = gisele_folder+'\Case studies/' + case_study+'/Input/MILP/all_data/'
    write = gisele_folder+'\Case studies/'+ case_study+'\Input\MILP\MILP_processed/'
    Connections=pd.read_csv(read_output+'connections_output.csv')
    All_connections_dist=pd.read_csv(read_input+'distances.csv')
    #All_connections_cost=pd.read_csv(read_dir+'weights.csv')
    Voltages=pd.read_csv(read_output+'Voltages.csv')
    Nodes=pd.read_csv(read_general_data+'All_Nodes.csv')
    Lines_clusters = pd.read_csv(read_output+'links_clusters.csv')
    Lines_clusters_original = pd.read_csv(read_general_data+'Lines_clusters.csv')
    Lines_clusters_marked = gpd.read_file(read_general_data+'Lines_marked/Lines_marked.shp')
    Lines_clusters_marked['Conn_param'] = Lines_clusters_marked['Conn_param'].astype(int)

    #Lines=pd.DataFrame(columns=[['id1','id2','Length','Cost','geom']])
    try:
        Microgrid=pd.read_csv(read_output+'Microgrid.csv')
    except:
        print('Only on-grid')
    try:
        dist_from_PS=pd.read_csv(read_output+'Distances.csv')
        case = 'reliability'
    except:
        case = 'no_reliability'

    if 'Type' in Lines_clusters.columns:
        case2 = 'Multiple_Types'
    else:
        case2 = 'One_Type'
    id1=[]
    id2=[]
    Length=[]
    Cost=[]
    geom=[]
    Loading=[]
    Type=[]
    Segments_connections = gpd.read_file(read_general_data +'Lines_connections/Lines_connections.shp')
    Segments_connections['ID2']=Segments_connections['ID2'].astype(int)
    Segments_connections['ID1']=Segments_connections['ID1'].astype(int)
    for index, row in Connections.iterrows():
        try:
            #geom.append(LineString(
            #    [(Nodes.loc[Nodes['ID'] == int(row['id1']), 'X'], Nodes.loc[Nodes['ID'] == int(row['id1']), 'Y']),
            #     (Nodes.loc[Nodes['ID'] == int(row['id2']), 'X'], Nodes.loc[Nodes['ID'] == int(row['id2']), 'Y'])]))
            linestring  = Segments_connections.loc[(Segments_connections['ID1']==row['id1']) & (Segments_connections['ID2']==row['id2']),'geometry']
            try:
                linestring=linestring.values[0]
            except:
                linestring = Segments_connections.loc[(Segments_connections['ID1'] == row['id2']) & (Segments_connections['ID2'] == row['id1']), 'geometry'].values[0]
            geom.append(linestring)
            id1.append(row['id1'])
            id2.append(row['id2'])
            if case2=='Multiple_Types':
                Type.append(row['Type'])
            Loading.append(round(row['power'],3))
            distance=All_connections_dist.loc[(All_connections_dist['ID1']==row['id1']) & (All_connections_dist['ID2']==row['id2']),'Length']
            #cost= All_connections_cost.loc[(All_connections_cost['id1']==row['id1']) & (All_connections_cost['id2']==row['id2']),'distance']
            Length.append(round(abs(float(distance)),3))
            #Cost.append(float(cost))


        except:
            print('MG connection - fake node')
    #Data = {'id1': id1,'id2': id2, 'Length': Length, 'Cost': Cost, 'Loading': Loading, 'Geom': geom}

    try:
        if case2 == 'Multiple_Types':
            Data = {'id1': id1, 'id2': id2, 'Length': Length, 'Loading': Loading, 'Geom': geom ,'Type': Type}
            Lines = pd.DataFrame(Data)
            Lines.to_csv(write+'output_connections.csv',index=False)
        else:
            Data = {'id1': id1, 'id2': id2, 'Length': Length, 'Loading': Loading, 'Geom': geom}
            Lines = pd.DataFrame(Data)
            Lines.to_csv(write + 'output_connections.csv', index=False)
    except:
        print('empty file - all Microgrids')

    x=[]
    y=[]
    Power=[]
    voltage=[]
    ID=[]
    mg=[]
    cluster=[]
    dist=[]
    if not case == 'reliability':
        for index,row in Voltages.iterrows():
            try:
                id=row['index']

                x.append(float(Nodes.loc[(Nodes['ID']==id),'X']))
                y.append(float(Nodes.loc[(Nodes['ID']==id),'Y']))
                cluster.append(float(Nodes.loc[(Nodes['ID']==id),'Cluster']))
                ID.append(id)
                voltage.append(round(row['voltage [p.u]'],3))
                Power.append(round(float(Nodes.loc[(Nodes['ID']==id),'MV_Power']),3))
            except:
                print('microgrid fake node')
                print(row['index'])
        Data = {'ID': ID, 'X': x, 'Y': y, 'Voltage': voltage, 'Power': Power, 'Cluster': cluster}
    else:
        for index, row in Voltages.iterrows():
            try:
                id = row['index']

                x.append(float(Nodes.loc[(Nodes['ID'] == id), 'X']))
                y.append(float(Nodes.loc[(Nodes['ID'] == id), 'Y']))
                cluster.append(float(Nodes.loc[(Nodes['ID'] == id), 'Cluster']))
                ID.append(id)
                voltage.append(round(row['voltage [p.u]'],3))
                Power.append(round(float(Nodes.loc[(Nodes['ID']==id),'MV_Power']),3))
                dist.append(round(-float(dist_from_PS.loc[(dist_from_PS['index'] == id), 'length[m]']),3))
            except:
                print('microgrid fake node')
                print(row['index'])
        Data = {'ID': ID, 'X': x, 'Y': y, 'Voltage': voltage, 'Power': Power, 'Cluster': cluster,'Distance':dist}

    #Data={'ID': ID, 'X': x, 'Y': y, 'Voltage': voltage,'Microgrid': mg,'Power': Power }
    #Data={'ID': ID, 'X': x, 'Y': y, 'Voltage': voltage,'Microgrid': mg,'Power': Power, 'dist_from_PS': dist}
    Nodes=pd.DataFrame(Data)
    Nodes.to_csv(write+'output_nodes.csv',index=False)

    id1=[]
    id2=[]
    Length=[]
    Cost=[]
    geom=[]
    Loading=[]
    Type=[]
    Lines_clusters=Lines_clusters[Lines_clusters['power']!=0]
    for index, row in Lines_clusters.iterrows():
        id1.append(row['id1'])
        id2.append(row['id2'])
        Loading.append(round(row['power'],3))
        distance=All_connections_dist.loc[(All_connections_dist['ID1']==row['id1']) & (All_connections_dist['ID2']==row['id2']),'Length']
        if case2 == 'Multiple_Types':
            Type.append(row['Type'])
        #cost= All_connections_cost.loc[(All_connections_cost['id1']==row['id1']) & (All_connections_cost['id2']==row['id2']),'distance']
        Length.append(round(abs(float(distance)),3))
        #Cost.append(float(cost))

        # locate the conn param that connects the output of the MILP with the actual line following the roads
        conn_param = Lines_clusters_original.loc[(Lines_clusters_original['ID1'] == row['id1']) &
                                            (Lines_clusters_original['ID2'] == row['id2']), 'Conn_param']
        try:
            conn_param=int(conn_param.values[0])
        except:
            conn_param = int(Lines_clusters_original.loc[(Lines_clusters_original['ID1'] == row['id2']) &
                                                     (Lines_clusters_original['ID2'] == row['id1']), 'Conn_param'].values[0])
        print(conn_param)
        # now we have the conn_param, we should access the file Lines_clusters_marked
        LiNeS = Lines_clusters_marked.loc[Lines_clusters_marked['Conn_param'] == conn_param, 'geometry']
        geometry = MultiLineString([line for line in LiNeS])
        geom.append(geometry)
    if case2 == 'Multiple_Types':
        Data = {'id1': id1, 'id2': id2, 'Length': Length, 'Loading': Loading, 'Geom': geom,'Type':Type}
    else:
        Data = {'id1': id1, 'id2': id2, 'Length': Length, 'Loading': Loading, 'Geom': geom}
    Lines = pd.DataFrame(Data)
    Lines.to_csv(write + 'output_cluster_lines.csv', index=False)
