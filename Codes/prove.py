import os
import pandas as pd

clusters_list_2 = [0, 1, 2, 3]

total_energy = pd.DataFrame(index= clusters_list_2, columns=['Total Energy Consumption [kWh]'])
os.chdir(r'..//Input//7_sizing_calliope//timeseries_data')


load_ref = pd.read_csv('demand_power_ref.csv', index_col=0, header=0)
load_cluster = load_ref.copy()

load_cluster['X1'] = load_ref['X1'] / 706 * 1000
energy = -1 * load_cluster.values.sum()
total_energy.at[0, 'Total Energy Consumption [kWh]'] = energy

print('energy:')
print(energy)





