import pandas as pd


def emission_factor(country):
    energy_mix = pd.read_csv(
        'Input/Datasets/Emission/country_electricity_mix.csv')
    emission_factors = pd.read_csv(
        'Input/Datasets/Emission/emission_factor.csv',index_col='fuel') #ton CO2/MWh

    energy_mix = energy_mix[energy_mix['Country'] == country]
    energy_mix = energy_mix[energy_mix['Year'].isin([2017,2018,2019])]
    energy_mix = energy_mix.groupby('Variable').mean()
    country_emission_factor = energy_mix.loc['Coal', 'Value (TWh)'] * emission_factors.loc['Coal','emission_factor']/emission_factors.loc['Coal','efficiency']+\
        energy_mix.loc['Gas', 'Value (TWh)'] * emission_factors.loc['Gas','emission_factor']/emission_factors.loc['Gas','efficiency']+\
        energy_mix.loc['Biomass and waste', 'Value (TWh)'] * emission_factors.loc['Wood','emission_factor']/emission_factors.loc['Wood','efficiency']+\
        energy_mix.loc['Other fossil', 'Value (TWh)'] * emission_factors.loc['Oil','emission_factor']/emission_factors.loc['Oil','efficiency']
    country_specific_emission_factor = country_emission_factor/energy_mix.loc['Production', 'Value (TWh)']*1000 #kg CO2/MWh

    return country_specific_emission_factor




