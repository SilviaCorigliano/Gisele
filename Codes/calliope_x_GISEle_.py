import calliope
import os
import pandas as pd
import numpy as np


# setting the path for calliope

# calling and running the Model
calliope.set_log_level('INFO')
# model = calliope.Model(os.path.join(_PATHS['GISEle'], 'model.yaml'), scenario='time_clustering')
model = calliope.Model(os.path.join(_PATHS['GISEle'], 'model.yaml'))
model.run()
print("Model runned")
c = 0
os.chdir(r'..//Output//Sizing_calliope')

'Saving results'
model.to_csv("GISEle_results_" + str(c))
model.plot.summary(to_file='gisele' + str(c) + '.html')

'Extracting all technologies'
os.chdir("GISEle_results_" + str(c))
inputs_techs = pd.read_csv('inputs_colors.csv', header=None, index_col=0)
technologies = inputs_techs.index.values
technologies = technologies[technologies != 'demand_electricity']
print(technologies)

'Finding lifetimes'
title = 'Lifetime [years]'
inputs_lifetime = pd.read_csv('inputs_lifetime.csv', header=None, index_col=1)
lifetimes = pd.DataFrame(index=technologies, columns=[title])

for tech_given in inputs_lifetime.index.values:
    if tech_given in technologies:
        life = inputs_lifetime.loc[str(tech_given), 2]
        lifetimes.at[tech_given, title] = life
print(lifetimes)

'Finding which of the technologies is being used in the results and their corresponding capacities'
title = 'Capacity [kW]'
results_energy_cap = pd.read_csv('results_energy_cap.csv', header=None, index_col=1)
capacities = pd.DataFrame(index=technologies, columns=[title])

for tech_given in inputs_lifetime.index.values:
    if tech_given in technologies:
        cap = results_energy_cap.loc[str(tech_given), 2]
        capacities.at[tech_given, title] = cap
print(capacities)

'Finding which of the STORAGE technologies is being used in the results and their corresponding capacities'
title = 'Capacity [kWh]'
results_storage_cap = pd.read_csv('results_storage_cap.csv', header=None, index_col=1)

storage_cap = []
storage_used = []

for tech_used in results_storage_cap.index.values:
    if tech_used in technologies:
        co_sto = float(results_storage_cap.loc[str(tech_used), 2])
        storage_cap.append(round(co_sto, 2))
        storage_used.append(tech_used)

sto_capacities = pd.DataFrame(index=storage_used, data=storage_cap, columns=[title])
print(sto_capacities)

'Creating the costs table for all components over the time'
costs_rows = []
for tech in technologies:
    costs_rows.append("CAPEX_" + str(tech))
    costs_rows.append("OPEX_" + str(tech))
undiscounted_costs = pd.DataFrame(columns=time_span, index=costs_rows)
print(undiscounted_costs)

'Extracting specific costs €/kW'
title = 'Specific cost [€/kW]'
inputs_cost_energy_cap = pd.read_csv('inputs_cost_energy_cap.csv', header=None, index_col=1)
specific_costs = pd.DataFrame(index=technologies, columns=[title])
for tech_used in inputs_cost_energy_cap.index.values:
    if tech_used in technologies:
        spec_cost = inputs_cost_energy_cap.loc[str(tech_used), 3]
        specific_costs.at[tech_used, title] = spec_cost
print(specific_costs)

'Extracting specific costs €/kWh for storage only'
title = 'Specific cost [€/kWh]'
inputs_cost_storage_cap = pd.read_csv('inputs_cost_storage_cap.csv', header=None, index_col=1)
specific_cost_storage = []
storage_techs = []
# ----------------------------------------------------

'Evaluating OPEX'
os.chdir(r'..//..//..//Input//5_components')
om_costs = pd.read_csv('om_costs.csv', index_col=0, delimiter=';', header=0)
print(om_costs)
os.chdir(r'..//..//Output//Sizing_calliope')
# ----------------------------------------------------

for tech_used in inputs_cost_storage_cap.index.values:
    if tech_used in technologies:
        co_sto = float(inputs_cost_storage_cap.loc[str(tech_used), 3])
        specific_cost_storage.append(round(co_sto, 2))
        storage_techs.append(tech_used)
specific_costs_storage = pd.DataFrame(index=storage_techs, data=specific_cost_storage, columns=[title])
print(specific_costs_storage)
# ----------------------------------------------------------------

grid_resume = pd.read_csv('grid_resume.csv', header=0, delimiter=',', index_col=0)

resume = pd.concat([lifetimes, capacities, sto_capacities, specific_costs,
                    specific_costs_storage, om_costs['%_as_fraction/year'], om_costs['€/year']], axis=1, sort=False)

'Only for lines'
resume.at['power_lines', 'total investment_cost [€]'] = grid_resume.loc[c, 'Grid_Cost']

resume = resume.fillna(0)
resume['investment_cost_per kW [€]'] = resume['Capacity [kW]'] * resume['Specific cost [€/kW]']
resume['investment_cost_per kWh [€]'] = resume['Capacity [kWh]'] * resume['Specific cost [€/kWh]']
resume['total investment_cost [€]'] = resume['investment_cost_per kWh [€]'] + resume['investment_cost_per kW [€]']
resume['total_om_costs_as_fraction [€/year]'] = resume['total investment_cost [€]'] * resume[
    '%_as_fraction/year'] / 100

print(resume)
resume.to_csv('resume.csv')

'Inserting CAPEX'

for capex in undiscounted_costs.index.values:
    for techs in technologies:
        if techs in capex:
            if 'CAPEX' in capex:
                investment_cost = resume.at[techs, 'total investment_cost [€]']
                undiscounted_costs.at[capex, 0] = investment_cost
                if resume.at[techs, 'Lifetime [years]'] != 0 and resume.at[
                    techs, 'Lifetime [years]'] < project_life:
                    n_replacements = int(project_life / resume.at[techs, 'Lifetime [years]'] - 0.5)
                    for n in range(1, n_replacements + 1):
                        undiscounted_costs.at[capex, n] = investment_cost
############        residual_value =


'Inserting OPEX'

for opex in undiscounted_costs.index.values:
    for techs in technologies:
        if techs in opex:
            if 'OPEX' in opex:
                om_cost_as_fraction = resume.loc[techs, 'total_om_costs_as_fraction [€/year]']
                om_cost_fix = om_costs.loc[techs, '€/year']
                total_om_cost = om_cost_as_fraction + om_cost_fix
                for y in range(1, project_life):
                    undiscounted_costs.at[opex, y] = total_om_cost

undiscounted_costs = undiscounted_costs.fillna(0)
print(undiscounted_costs)

# discount_rate = float(input("What's the discount rate?: [%]"))
discount_rate = 10 / 100
discount_factors = []
for t in time_span:
    disc = 1 / ((1 + discount_rate) ** t)
    discount_factors.append(disc)

discounted_costs = undiscounted_costs * discount_factors
undiscounted_costs.to_csv('undiscounted_costs.csv')
discounted_costs.to_csv('discounted_costs.csv')