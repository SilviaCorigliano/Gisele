''' ALL DIFFERENT MILP MODELS IN ONE FILE
1. MILP WITHOUT MG -> Doesn't consider the Microgrid option and works with only 1 type of cable. However, it considers reliability.
2. MILP2 -> Consider
3. MILP3
4. MILP4
5. MILP5
'''

from __future__ import division

from pyomo.opt import SolverFactory
from pyomo.core import AbstractModel
from pyomo.dataportal.DataPortal import DataPortal
from pyomo.environ import *
import pandas as pd
from datetime import datetime
import os
def MILP_without_MG(gisele_folder,case_study,n_clusters,coe,voltage,resistance,reactance,Pmax,line_cost):
    ############ Create abstract model ###########
    model = AbstractModel()
    data = DataPortal()
    MILP_input_folder = gisele_folder + '/Case studies/' + case_study + '/Input/MILP/MILP_input'
    MILP_output_folder = gisele_folder + '/Case studies/' + case_study + '/Input/MILP/MILP_output'
    os.chdir(MILP_input_folder)

    # Define some basic parameter for the per unit conversion and voltage limitation
    Abase = 1
    Vmin = 0.9

    ####################Define sets#####################
    model.N = Set()
    data.load(filename='nodes.csv', set=model.N)  # first row is not read
    model.N_clusters = Set()
    data.load(filename='nodes_clusters.csv', set=model.N_clusters)
    model.N_PS = Set()
    data.load(filename='nodes_PS.csv', set=model.N_PS)
    # Node corresponding to primary substation

    # Allowed connections
    model.links = Set(dimen=2)  # in the csv the values must be delimited by commas
    data.load(filename='links_all.csv', set=model.links)

    model.links_clusters = Set(dimen=2)
    data.load(filename='links_clusters.csv', set=model.links_clusters)

    model.links_decision = Set(dimen=2)
    data.load(filename='links_decision.csv', set=model.links_decision)

    # Nodes are divided into two sets, as suggested in https://pyomo.readthedocs.io/en/stable/pyomo_modeling_components/Sets.html:
    # NodesOut[nodes] gives for each node all nodes that are connected to it via outgoing links
    # NodesIn[nodes] gives for each node all nodes that are connected to it via ingoing links

    def NodesOut_init(model, node):
        retval = []
        for (i, j) in model.links:
            if i == node:
                retval.append(j)
        return retval

    model.NodesOut = Set(model.N, initialize=NodesOut_init)

    def NodesIn_init(model, node):
        retval = []
        for (i, j) in model.links:
            if j == node:
                retval.append(i)
        return retval

    model.NodesIn = Set(model.N, initialize=NodesIn_init)

    #####################Define parameters#####################

    # Electric power in the nodes (injected (-) or absorbed (+))
    model.Psub = Param(model.N_clusters)
    data.load(filename='power_nodes.csv', param=model.Psub)

    model.ps_cost = Param(model.N_PS)
    data.load(filename='PS_costs.csv', param=model.ps_cost)

    model.PSmax = Param(model.N_PS)
    data.load(filename='PS_power_max.csv', param=model.PSmax)

    model.PS_voltage = Param(model.N_PS)
    data.load(filename='PS_voltage.csv', param=model.PS_voltage)

    model.PS_distance = Param(model.N_PS)
    data.load(filename='PS_distance.csv', param=model.PS_distance)

    # Connection distance of all the edges
    model.dist = Param(model.links)
    data.load(filename='distances.csv', param=model.dist)

    model.weights = Param(model.links_decision)
    data.load(filename='weights_decision_lines.csv', param=model.weights)
    # Electrical parameters of all the cables
    model.V_ref = Param(initialize=voltage)
    model.A_ref = Param(initialize=Abase)
    model.E_min = Param(initialize=Vmin)

    model.R_ref = Param(initialize=resistance)
    model.X_ref = Param(initialize=reactance)
    model.P_max = Param(initialize=Pmax)
    model.cf = Param(initialize=line_cost)

    model.Z = Param(initialize=model.R_ref + model.X_ref * 0.5)
    model.Z_ref = Param(initialize=model.V_ref ** 2 / Abase)

    model.n_clusters = Param(initialize=n_clusters)
    model.coe = Param(initialize=coe)

    #####################Define variables#####################

    # binary variable x[i,j]: 1 if the connection i,j is present, 0 otherwise
    model.x = Var(model.links_decision, within=Binary)
    # power[i,j] is the power flow of connection i-j
    model.P = Var(model.links)
    # positive variables E(i) is p.u. voltage at each node
    model.E = Var(model.N, within=NonNegativeReals)

    # binary variable k[i]: 1 if node i is a primary substation, 0 otherwise
    model.k = Var(model.N_PS, within=Binary)
    # Power output of Primary substation
    model.PPS = Var(model.N_PS, within=NonNegativeReals)
    model.positive_p = Var(model.links_clusters, within=Binary)
    model.Distance = Var(model.N, within=Reals)

    model.cable_type = Var(model.links)

    #####################Define constraints###############################

    # Radiality constraint
    def Radiality_rule(model):
        return summation(model.x) == model.n_clusters

    model.Radiality = Constraint(rule=Radiality_rule)

    # Power flow constraints
    def Power_flow_conservation_rule(model, node):
        return (sum(model.P[j, node] for j in model.NodesIn[node]) - sum(
            model.P[node, j] for j in model.NodesOut[node])) == model.Psub[node]

    model.Power_flow_conservation = Constraint(model.N_clusters, rule=Power_flow_conservation_rule)

    def Power_flow_conservation_rule3(model, node):
        return (sum(model.P[j, node] for j in model.NodesIn[node]) - sum(
            model.P[node, j] for j in model.NodesOut[node])) == - model.PPS[node]

    model.Power_flow_conservation3 = Constraint(model.N_PS, rule=Power_flow_conservation_rule3)

    def Power_upper_decision(model, i, j):
        return model.P[i, j] <= model.P_max * model.x[i, j]

    model.Power_upper_decision = Constraint(model.links_decision, rule=Power_upper_decision)

    def Power_lower_decision(model, i, j):
        return model.P[i, j] >= 0  # -model.P_max*model.x[i,j]

    model.Power_lower_decision = Constraint(model.links_decision, rule=Power_lower_decision)

    def Power_upper_clusters(model, i, j):
        return model.P[i, j] <= model.P_max * model.positive_p[i, j]

    model.Power_upper_clusters = Constraint(model.links_clusters, rule=Power_upper_clusters)

    def Power_lower_clusters(model, i, j):
        return model.P[i, j] >= 0  # -model.P_max*model.x[i,j]

    model.Power_lower_clusters = Constraint(model.links_clusters, rule=Power_lower_clusters)

    # Voltage constraints
    def Voltage_balance_rule(model, i, j):
        return (model.E[i] - model.E[j]) + model.x[i, j] - 1 <= model.dist[i, j] / 1000 * model.P[
            i, j] * model.Z / model.Z_ref

    model.Voltage_balance_rule = Constraint(model.links_decision, rule=Voltage_balance_rule)

    def Voltage_balance_rule2(model, i, j):
        return (model.E[i] - model.E[j]) - model.x[i, j] + 1 >= model.dist[i, j] / 1000 * model.P[
            i, j] * model.Z / model.Z_ref

    model.Voltage_balance_rule2 = Constraint(model.links_decision, rule=Voltage_balance_rule2)

    def Voltage_balance_rule3(model, i, j):
        return (model.E[i] - model.E[j]) + model.positive_p[i, j] - 1 <= model.dist[i, j] / 1000 * model.P[
            i, j] * model.Z / model.Z_ref

    model.Voltage_balance_rule3 = Constraint(model.links_clusters, rule=Voltage_balance_rule3)

    def Voltage_balance_rule4(model, i, j):
        return (model.E[i] - model.E[j]) - model.positive_p[i, j] + 1 >= model.dist[i, j] / 1000 * model.P[
            i, j] * model.Z / model.Z_ref

    model.Voltage_balance_rule4 = Constraint(model.links_clusters, rule=Voltage_balance_rule4)

    def Voltage_limit(model, i):
        return model.E[i] >= model.k[i] * (model.PS_voltage[i] - model.E_min) + model.E_min

    model.Voltage_limit = Constraint(model.N_PS, rule=Voltage_limit)

    def Voltage_PS2(model, i):
        return model.E[i] <= model.PS_voltage[i]

    model.Voltage_PS2 = Constraint(model.N_PS, rule=Voltage_PS2)

    def Voltage_limit_clusters2(model, i):
        return model.E[i] >= model.E_min

    model.Voltage_limit_clusters2 = Constraint(model.N_clusters, rule=Voltage_limit_clusters2)

    def PS_power_rule_upper(model, i):
        return model.PPS[i] <= model.PSmax[i] * model.k[i]

    model.PS_power_upper = Constraint(model.N_PS, rule=PS_power_rule_upper)

    def distance_from_PS(model, i):
        return model.Distance[i] <= -model.PS_distance[i]

    model.distance_from_PS = Constraint(model.N_PS, rule=distance_from_PS)

    def distance_from_PS2(model, i):
        return model.Distance[i] >= (model.k[i] - 1) * 200 - model.PS_distance[i] * model.k[i]

    model.distance_from_PS2 = Constraint(model.N_PS, rule=distance_from_PS2)

    def distance_balance_decision(model, i, j):
        return model.Distance[i] - model.Distance[j] + 1000 * (model.x[i, j] - 1) <= model.dist[i, j] / 1000

    model.distance_balance_decision = Constraint(model.links_decision, rule=distance_balance_decision)

    def distance_balance_decision2(model, i, j):
        return (model.Distance[i] - model.Distance[j]) - 1000 * (model.x[i, j] - 1) >= model.dist[i, j] / 1000

    model.distance_balance_decision2 = Constraint(model.links_decision, rule=distance_balance_decision2)

    def distance_balance_clusters(model, i, j):
        return model.Distance[i] - model.Distance[j] + 1000 * (model.positive_p[i, j] - 1) <= model.dist[i, j] / 1000

    model.distance_balance_clusters = Constraint(model.links_clusters, rule=distance_balance_clusters)

    def distance_balance_clusters2(model, i, j):
        return (model.Distance[i] - model.Distance[j]) - 1000 * (model.positive_p[i, j] - 1) >= model.dist[i, j] / 1000

    model.distance_balance_clusters2 = Constraint(model.links_clusters, rule=distance_balance_clusters2)

    def Balance_rule(model):
        return (sum(model.PPS[i] for i in model.N_PS) - sum(model.Psub[i] for i in model.N_clusters)) == 0

    model.Balance = Constraint(rule=Balance_rule)

    def anti_paralel(model, i, j):
        return model.x[i, j] + model.x[j, i] <= 1

    model.anti_paralel = Constraint(model.links_decision, rule=anti_paralel)

    def anti_paralel_clusters(model, i, j):
        return model.positive_p[i, j] + model.positive_p[j, i] == 1

    model.anti_paralel_clusters = Constraint(model.links_clusters, rule=anti_paralel_clusters)
    ####################Define objective function##########################

    ####################Define objective function##########################
    reliability_index = 1000

    def ObjectiveFunction(model):
        return summation(model.weights, model.x) * model.cf / 1000 + summation(model.ps_cost, model.k)

        # return summation(model.weights, model.x) * model.cf / 1000  + summation(model.ps_cost,model.k) - sum(model.Psub[i]*model.Distance[i] for i in model.N_clusters)*reliability_index

    model.Obj = Objective(rule=ObjectiveFunction, sense=minimize)

    #############Solve model##################

    instance = model.create_instance(data)
    print('Instance is constructed:', instance.is_constructed())
    # opt = SolverFactory('cbc',executable=r'C:\Users\Asus\Desktop\POLIMI\Thesis\GISELE\Gisele_MILP\cbc')
    opt = SolverFactory('gurobi')
    # opt.options['numericfocus']=0
    # opt.options['mipgap'] = 0.0002
    # opt.options['presolve']=2
    # opt.options['mipfocus']=2
    # opt = SolverFactory('cbc',executable=r'C:\Users\Asus\Desktop\POLIMI\Thesis\GISELE\New folder\cbc')
    print('Starting optimization process')
    time_i = datetime.now()
    opt.solve(instance, tee=True, symbolic_solver_labels=True)
    time_f = datetime.now()
    print('Time required for optimization is', time_f - time_i)
    links = instance.x
    power = instance.P
    subs = instance.k
    voltage = instance.E
    PS = instance.PPS
    DISTANCE = instance.Distance

    links_clusters = instance.links_clusters
    # voltage_drop=instance.z
    connections_output = pd.DataFrame(columns=[['id1', 'id2', 'power']])
    PrSubstation = pd.DataFrame(columns=[['index', 'power']])
    all_lines = pd.DataFrame(columns=[['id1', 'id2', 'power']])
    Voltages = pd.DataFrame(columns=[['index', 'voltage [p.u]']])
    Links_Clusters = pd.DataFrame(columns=[['id1', 'id2', 'power']])
    distance = pd.DataFrame(columns=[['index', 'length[km]']])
    k = 0
    for index in links:
        if int(round(value(links[index]))) == 1:
            connections_output.loc[k, 'id1'] = index[0]
            connections_output.loc[k, 'id2'] = index[1]
            connections_output.loc[k, 'power'] = value(power[index])
            k = k + 1
    k = 0
    for index in subs:
        if int(round(value(subs[index]))) == 1:
            PrSubstation.loc[k, 'index'] = index
            PrSubstation.loc[k, 'power'] = value(PS[index])
            print(((value(PS[index]))))
            k = k + 1
    k = 0
    for v in voltage:
        Voltages.loc[k, 'index'] = v
        Voltages.loc[k, 'voltage [p.u]'] = value(voltage[v])
        k = k + 1
    k = 0
    for index in power:
        all_lines.loc[k, 'id1'] = index[0]
        all_lines.loc[k, 'id2'] = index[1]
        all_lines.loc[k, 'power'] = value(power[index])
        k = k + 1
    k = 0

    for index in links_clusters:
        Links_Clusters.loc[k, 'id1'] = index[0]
        Links_Clusters.loc[k, 'id2'] = index[1]
        Links_Clusters.loc[k, 'power'] = value(power[index])
        k = k + 1
    k = 0
    for dist in DISTANCE:
        distance.loc[k, 'index'] = dist
        distance.loc[k, 'length[m]'] = value(DISTANCE[dist])
        k = k + 1
    Links_Clusters.to_csv(MILP_output_folder + '/links_clusters.csv', index=False)
    connections_output.to_csv(MILP_output_folder + '/connections_output.csv', index=False)
    PrSubstation.to_csv(MILP_output_folder + '/PrimarySubstations.csv', index=False)
    Voltages.to_csv(MILP_output_folder + '/Voltages.csv', index=False)
    all_lines.to_csv(MILP_output_folder + '/all_lines.csv', index=False)
    distance.to_csv(MILP_output_folder + '/Distances.csv', index=False)