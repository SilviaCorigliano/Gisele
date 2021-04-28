import pandas as pd


def emission_factor(country, emission_type):
    energy_mix = pd.read_csv(
        'Input/Datasets/Emission/country_electricity_mix.csv')
    emission_factors = pd.read_csv(
        'Input/Datasets/Emission/emission_factor.csv',index_col='fuel') # ton CO2/MWh
    if emission_type=='direct':
        column='dir_em'
    else:
        column='indir_em'
    energy_mix = energy_mix[energy_mix['Country'] == country]
    energy_mix = energy_mix[energy_mix['Year'].isin([2017,2018,2019])]
    energy_mix = energy_mix.groupby('Variable').mean()
    country_emission_factor = energy_mix.loc['Coal', 'Value (TWh)'] * emission_factors.loc['Coal',column]/emission_factors.loc['Coal','efficiency']+\
        energy_mix.loc['Gas', 'Value (TWh)'] * emission_factors.loc['Gas',column]/emission_factors.loc['Gas','efficiency']+\
        energy_mix.loc['Biomass and waste', 'Value (TWh)'] * emission_factors.loc['Wood',column]/emission_factors.loc['Wood','efficiency']+\
        energy_mix.loc['Other fossil', 'Value (TWh)'] * emission_factors.loc['Oil',column]/emission_factors.loc['Oil','efficiency']+\
        energy_mix.loc['Hydro','Value (TWh)'] * emission_factors.loc['Hydro',column]+\
        energy_mix.loc['Solar','Value (TWh)']* emission_factors.loc['PV',column]+ \
        energy_mix.loc['Wind', 'Value (TWh)']* emission_factors.loc['Wind',column]+ \
        energy_mix.loc['Nuclear', 'Value (TWh)'] * emission_factors.loc['Nuclear', column]
    country_specific_emission_factor = country_emission_factor/energy_mix.loc['Production', 'Value (TWh)'] *1000 # kgCO2/MWh

    return country_specific_emission_factor


def reliability_grid(country):
    reliability = pd.read_csv(
        'Input/Datasets/Reliability/WB data_ enterprise survey_SSA.csv')
    saidi_grid = float(reliability.loc[reliability['Economy'] == country,
                                 'unavailability [h/month]'].values[0])*12 # [h/year]
    return saidi_grid

def line_reliability():
    rel_components = pd.read_csv('Input/Datasets/Reliability/reliability_components.csv',index_col='Unnamed: 0')
    line_rel=rel_components.loc['Line','unavailability [h/year]'] # h/km/year
    return line_rel

