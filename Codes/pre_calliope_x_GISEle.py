import yaml
import os
import pandas as pd
import glob
from data_import import *

def substitution(yaml_file, before, after):
    file = str(yaml_file) + '.yaml'
    #before = str(before)
    #after = str(after)
    new_file = []
    # Open template file
    with open(file, 'r') as f:
        # Iterate through each line
        for l in f:
            # Replace every {{id}} occurrence
            new_file.append(l.replace(before, after))
    # Save the new file
    with open(file, 'w') as f:
        for l in new_file:
            f.write(l)


def pre_calliope(i, project_life, lat, long):
    'Importing info'
    os.chdir(r'Input//5_components')

    demand_df = pd.read_csv('Demand.csv', header=0, index_col=0)
    converter_df = pd.read_csv('Converter.csv', header=0, index_col=0)
    generator_df = pd.read_csv('Generator.csv', header=0, index_col=0)
    hydro_df = pd.read_csv('Hydro.csv', header=0, index_col=0)
    MV_df = pd.read_csv('MV.csv', header=0, index_col=0)
    PV_df = pd.read_csv('PV.csv', header=0, index_col=0)
    storage_df = pd.read_csv('Storage.csv', header=0, index_col=0)
    wind_df = pd.read_csv('Wind.csv', header=0, index_col=0)

    components = [demand_df, converter_df, generator_df,
                  hydro_df, MV_df, PV_df, storage_df, wind_df]

    components_df = pd.concat([df.stack() for df in components], axis=0).unstack()

    components_df.to_csv('components.csv')

    components = pd.read_csv('components.csv', header=0, index_col=0)
    components.fillna(0)

    'Exporting yaml files'

    'Tech groups file'
    os.chdir(r'..//..//Input//7_sizing_calliope//model_config')


    '''info = 'electricity'
    interest_rate = 0.1
    techs_groups = dict(
        tech_groups=dict(
            supply=dict(essentials=dict(carrier_out=info),
                        costs=dict(monetary=dict(interest_rate=interest_rate))),
            conversion=dict(costs=dict(monetary=dict(interest_rate=interest_rate))),
            transmission=dict(costs=dict(monetary=dict(interest_rate=interest_rate))),
            storage=dict(costs=dict(monetary=dict(interest_rate=interest_rate)))
        )
    )
    
    with open('techs_groups.yaml', 'w') as outfile:
        yaml.dump(techs_groups, outfile, default_flow_style=False)
    
    substitution('techs_groups', '  conversion', '    conversion')
    substitution('techs_groups', '    costs', '        costs')
    substitution('techs_groups', '      monetary', '            monetary')
    substitution('techs_groups', '        interest_rate', '                interest_rate')
    substitution('techs_groups', '  storage', '    storage')
    substitution('techs_groups', '    costs', '        costs')
    substitution('techs_groups', '      monetary', '            monetary')
    substitution('techs_groups', '        interest_rate', '                interest_rate')
    substitution('techs_groups', '  conversion', '    conversion')
    substitution('techs_groups', '    costs', '        costs')
    substitution('techs_groups', '      monetary', '            monetary')
    substitution('techs_groups', '        interest_rate', '                interest_rate')
    substitution('techs_groups', '    essentials', '            resource')
    substitution('techs_groups', '      carrier_out', '            resource')
    substitution('techs_groups', '  transmission', '    transmission')
    substitution('techs_groups', '    costs', '        costs')
    substitution('techs_groups', '      monetary', '            monetary')
    substitution('techs_groups', '        interest_rate', '                interest_rate')'''

    'One yaml tech file for each technology'
    for technologies in components.columns.values:

        file = 'techs_' + technologies

        if 'PV' in technologies:

            n = str(technologies)

            name = components.loc['name', technologies]
            parent = components.loc['parent', technologies]
            carrier_out = components.loc['carrier_out', technologies]

            resource = str(components.loc['resource', technologies])
            resource_unit = str(components.loc['resource_unit', technologies])
            energy_eff = float(components.loc['energy_eff', technologies])
            energy_cap_max = components.loc['energy_cap_max', technologies]
            resource_area_max = components.loc['resource_area_max', technologies]
            force_resource = str(components.loc['force_resource', technologies])
            resource_area_per_energy_cap = float(components.loc['resource_area_per_energy_cap', technologies])
            lifetime = float(components.loc['lifetime', technologies])

            energy_cap_cost = float(components.loc['energy_cap', technologies])
            om_annual_investment_fraction = float(components.loc['om_annual_investment_fraction', technologies])

            techs = dict(
                techs=dict(
                    PV=dict(
                        essentials=dict(name=name, parent=parent, carrier_out=carrier_out),
                        constraints=dict(resource=resource, resource_unit=resource_unit,
                                         energy_eff=energy_eff,
                                         energy_cap_max=energy_cap_max, resource_area_max=resource_area_max,
                                         force_resource=force_resource,
                                         resource_area_per_energy_cap=resource_area_per_energy_cap, lifetime=lifetime),
                        costs=dict(monetary=dict(energy_cap=energy_cap_cost,
                                                 om_annual_investment_fraction=om_annual_investment_fraction))))
            )

            with open(file + '.yaml', 'w') as outfile:
                yaml.dump(techs, outfile, default_flow_style=False)

            substitution(file, '  PV:', '    ' + technologies + ':')
            substitution(file, '    constraints', '        constraints')
            substitution(file, '      energy_cap_max', '            energy_cap_max')
            substitution(file, '      energy_eff', '            energy_eff')
            substitution(file, '      force_resource', '            force_resource')
            substitution(file, '      lifetime', '            lifetime')
            substitution(file, '      resource_area_max', '            resource_area_max')
            substitution(file, '      resource_area_per_energy_cap', '            resource_area_per_energy_cap')
            substitution(file, '      resource_unit', '            resource_unit')
            substitution(file, '    costs', '        costs')
            substitution(file, '      monetary', '            monetary')
            substitution(file, '        energy_cap', '                energy_cap')
            substitution(file, '        om_annual_investment_fraction', '                om_annual_investment_fraction')
            substitution(file, '    essentials', '        essentials')
            substitution(file, '      carrier_out', '            carrier_out')
            substitution(file, '      name', '            name')
            substitution(file, '      parent', '            parent')
            substitution(file, "'TRUE'", 'true')
            substitution(file, '      resource', '            resource')

            substitution(file, '                  resource_area_per_energy_cap', '            resource_area_per_energy_cap')
            substitution(file, '                    energy_cap_max', '            energy_cap_max')
            substitution(file, '                  resource_area_max', '            resource_area_max')
            substitution(file, '                  resource_unit', '            resource_unit')



        elif 'Wind' in technologies:

            n = str(technologies)

            name = components.loc['name', technologies]
            parent = components.loc['parent', technologies]
            carrier_out = components.loc['carrier_out', technologies]

            resource = components.loc['resource', technologies]
            resource_unit = components.loc['resource_unit', technologies]
            energy_eff = float(components.loc['energy_eff', technologies])
            energy_cap_max = components.loc['energy_cap_max', technologies]
            energy_cap_per_unit = float(components.loc['energy_cap_per_unit', technologies])
            resource_area_max = components.loc['resource_area_max', technologies]
            force_resource = str(components.loc['force_resource', technologies])
            resource_area_per_energy_cap = float(components.loc['resource_area_per_energy_cap', technologies])
            lifetime = float(components.loc['lifetime', technologies])

            energy_cap_cost = float(components.loc['energy_cap', technologies])
            om_annual_investment_fraction = float(components.loc['om_annual_investment_fraction', technologies])

            techs = dict(
                techs=dict(
                    wind=dict(
                        essentials=dict(name=name, parent=parent, carrier_out=carrier_out),
                        constraints=dict(resource=resource, resource_unit=resource_unit,
                                         energy_eff=energy_eff, energy_cap_per_unit=energy_cap_per_unit,
                                         energy_cap_max=energy_cap_max, resource_area_max=resource_area_max,
                                         force_resource=force_resource, lifetime=lifetime),
                        costs=dict(monetary=dict(energy_cap=energy_cap_cost,
                                                 om_annual_investment_fraction=om_annual_investment_fraction))))
            )

            with open(file + '.yaml', 'w') as outfile:
                yaml.dump(techs, outfile, default_flow_style=False)

            substitution(file, '  wind:', '    ' + technologies + ':')
            substitution(file, '    constraints', '        constraints')
            substitution(file, '      energy_cap_max', '            energy_cap_max')
            substitution(file, '      energy_eff', '            energy_eff')
            substitution(file, '      force_resource', '            force_resource')
            substitution(file, '      lifetime', '            lifetime')
            substitution(file, '      resource_area_max', '            resource_area_max')
            substitution(file, '      resource_area_per_energy_cap', '            resource_area_per_energy_cap')
            substitution(file, '      resource_unit', '            resource_unit')
            substitution(file, '    costs', '        costs')
            substitution(file, '      monetary', '            monetary')
            substitution(file, '        energy_cap', '                energy_cap')
            substitution(file, '        om_annual_investment_fraction', '                om_annual_investment_fraction')
            substitution(file, '    essentials', '        essentials')
            substitution(file, '      carrier_out', '            carrier_out')
            substitution(file, '      name', '            name')
            substitution(file, '      parent', '            parent')
            substitution(file, "'TRUE'", 'true')
            substitution(file, '      resource', '            resource')
            substitution(file, '      energy_cap_per_unit', '            energy_cap_per_unit')

            substitution(file, '                  resource_area_per_energy_cap', '            resource_area_per_energy_cap')
            substitution(file, '                    energy_cap_max', '            energy_cap_max')
            substitution(file, '                  resource_area_max', '            resource_area_max')

            substitution(file, '                  resource_unit', '            resource_unit')

        elif 'Hydro' in technologies:

            n = str(technologies)

            name = components.loc['name', technologies]
            parent = components.loc['parent', technologies]
            carrier_out = components.loc['carrier_out', technologies]

            resource = components.loc['resource', technologies]
            resource_unit = components.loc['resource_unit', technologies]
            energy_eff = float(components.loc['energy_eff', technologies])
            energy_cap_max = float(components.loc['energy_cap_max', technologies])
            energy_cap_per_unit = float(components.loc['energy_cap_per_unit', technologies])
            resource_area_max = components.loc['resource_area_max', technologies]
            force_resource = str(components.loc['force_resource', technologies])
            lifetime = float(components.loc['lifetime', technologies])

            energy_cap_cost = float(components.loc['energy_cap', technologies])
            om_annual_investment_fraction = float(components.loc['om_annual_investment_fraction', technologies])

            techs = dict(
                techs=dict(
                    hydro=dict(
                        essentials=dict(name=name, parent=parent, carrier_out=carrier_out),
                        constraints=dict(resource=resource, resource_unit=resource_unit,
                                         energy_eff=energy_eff, energy_cap_per_unit=energy_cap_per_unit,
                                         energy_cap_max=energy_cap_max, resource_area_max=resource_area_max,
                                         force_resource=force_resource, lifetime=lifetime),
                        costs=dict(monetary=dict(energy_cap=energy_cap_cost,
                                                 om_annual_investment_fraction=om_annual_investment_fraction))))
            )

            with open(file + '.yaml', 'w') as outfile:
                yaml.dump(techs, outfile, default_flow_style=False)

            substitution(file, '  hydro:', '    ' + technologies + ':')
            substitution(file, '    constraints', '        constraints')
            substitution(file, '      energy_cap_max', '            energy_cap_max')
            substitution(file, '      energy_eff', '            energy_eff')
            substitution(file, '      force_resource', '            force_resource')
            substitution(file, '      lifetime', '            lifetime')
            substitution(file, '      resource_area_max', '            resource_area_max')
            substitution(file, '      resource_area_per_energy_cap', '            resource_area_per_energy_cap')
            substitution(file, '      resource_unit', '            resource_unit')
            substitution(file, '    costs', '        costs')
            substitution(file, '      monetary', '            monetary')
            substitution(file, '        energy_cap', '                energy_cap')
            substitution(file, '        om_annual_investment_fraction', '                om_annual_investment_fraction')
            substitution(file, '    essentials', '        essentials')
            substitution(file, '      carrier_out', '            carrier_out')
            substitution(file, '      name', '            name')
            substitution(file, '      parent', '            parent')
            substitution(file, "'TRUE'", 'true')
            substitution(file, '      resource', '            resource')
            substitution(file, '      energy_cap_per_unit', '            energy_cap_per_unit')
            substitution(file, '                  resource_area_per_energy_cap', '            resource_area_per_energy_cap')
            substitution(file, '                    energy_cap_max', '            energy_cap_max')
            substitution(file, '                  resource_area_max', '            resource_area_max')
            substitution(file, '                  resource_unit', '            resource_unit')




        elif 'Gen' in technologies:

            n = str(technologies)

            name = components.loc['name', technologies]
            parent = components.loc['parent', technologies]
            carrier_out = components.loc['carrier_out', technologies]

            resource = components.loc['resource', technologies]
            resource_unit = components.loc['resource_unit', technologies]
            energy_eff = float(components.loc['energy_eff', technologies])
            energy_cap_max = components.loc['energy_cap_max', technologies]
            energy_cap_per_unit = components.loc['energy_cap_per_unit', technologies]
            energy_ramping = components.loc['energy_ramping', technologies]
            resource_area_max = components.loc['resource_area_max', technologies]
            lifetime = float(components.loc['lifetime', technologies])

            energy_cap_cost = float(components.loc['energy_cap', technologies])
            om_annual_investment_fraction = float(components.loc['om_annual_investment_fraction', technologies])
            om_prod = float(components.loc['om_prod', technologies])

            techs = dict(
                techs=dict(
                    diesel=dict(
                        essentials=dict(name=name, parent=parent, carrier_out=carrier_out),
                        constraints=dict(resource=resource, resource_unit=resource_unit,
                                         energy_eff=energy_eff, energy_cap_per_unit=energy_cap_per_unit,
                                         energy_cap_max=energy_cap_max, resource_area_max=resource_area_max,
                                         energy_ramping=energy_ramping, lifetime=lifetime),
                        costs=dict(monetary=dict(energy_cap=energy_cap_cost,
                                                 om_annual_investment_fraction=om_annual_investment_fraction,
                                                 om_prod=om_prod))))
            )

            substitution(file, '  diesel:', '    ' + technologies + ':')

        elif 'V_' in technologies:

            n = str(technologies)

            name = components.loc['name', technologies]
            parent = components.loc['parent', technologies]
            carrier = components.loc['carrier', technologies]

            energy_eff = float(components.loc['energy_eff', technologies])
            energy_cap_max = components.loc['energy_cap_max', technologies]
            lifetime = float(components.loc['lifetime', technologies])

            energy_cap_per_distance = float(components.loc['energy_cap_per_distance', technologies])
            om_annual_investment_fraction = float(components.loc['om_annual_investment_fraction', technologies])

            techs = dict(
                techs=dict(
                    power_lines=dict(
                        essentials=dict(name=name, parent=parent, carrier=carrier),
                        constraints=dict(energy_eff=energy_eff, energy_cap_max=energy_cap_max, lifetime=lifetime),
                        costs=dict(monetary=dict(energy_cap_per_distance=energy_cap_per_distance,
                                                 om_annual_investment_fraction=om_annual_investment_fraction))))
            )
            with open(file + '.yaml', 'w') as outfile:
                yaml.dump(techs, outfile, default_flow_style=False)

            substitution(file, '  power_lines:', '    ' + technologies + ':')
            substitution(file, '    constraints', '        constraints')
            substitution(file, '      energy_cap_max', '            energy_cap_max')
            substitution(file, '      energy_eff', '            energy_eff')
            substitution(file, '      lifetime', '            lifetime')
            substitution(file, '    costs', '        costs')
            substitution(file, '      monetary', '            monetary')
            substitution(file, '        energy_cap_per_distance', '                energy_cap_per_distance')
            substitution(file, '        om_annual_investment_fraction', '                om_annual_investment_fraction')
            substitution(file, '    essentials', '        essentials')
            substitution(file, '      carrier', '            carrier')
            substitution(file, '      name', '            name')
            substitution(file, '      parent', '            parent')


    # --------------------------------------------------------------------------------------------- Model
    all_techs_and_locations = glob.glob('*.{}'.format('yaml'))
    all_imports = []
    for files in all_techs_and_locations:
        all_imports.append(['model_config/' + files])

    os.chdir(r'..')

    model = dict(
        import_=all_imports,
        model=dict(name='GISEle Model', calliope_version='0.6.3', timeseries_data_path='timeseries_data',
                   timeseries_dateformat='%d/%m/%Y %H:%M', random_seed=23,
                   time=dict(function='apply_clustering',
                             function_options=dict(clustering_func='kmeans', how='mean', k=12))),
        run=dict(mode='plan', solver='glpk', ensure_feasibility='true', bigM='1e6')
    )

    with open('model.yaml', 'w') as outfile:
        yaml.dump(model, outfile, default_flow_style=False)


    substitution('model', 'import_', 'import')
    substitution('model', '- - ', '    - ')

    substitution('model', '  calliope_version', '    calliope_version' )
    substitution('model', '  name', '    name' )
    substitution('model', '  random_seed', '    random_seed' )
    #substitution('model', '  subset_time', '    subset_time' )
    substitution('model', '  time', '    time' )
    substitution('model', '    function', '        function' )
    substitution('model', '    function_options', '        function_options' )
    substitution('model', '      clustering_func', '            clustering_func' )
    substitution('model', '      how', '            how' )
    substitution('model', '      k', '            k' )
    substitution('model', '  timeseries_data_path', '    timeseries_data_path' )
    substitution('model', '  timeseries_dateformat', '    timeseries_dateformat' )
    substitution('model', '  bigM', '    bigM' )
    substitution('model', '  ensure_feasibility', '    ensure_feasibility' )
    substitution('model', '  mode', '    mode' )
    substitution('model', '  solver', '    solver' )
    substitution('model', '            function_options', '        function_options' )
    substitution('model', '      timeseries_data_path', '    timeseries_data_path' )
    substitution('model', '      timeseries_dateformat', '    timeseries_dateformat' )

    os.chdir(r'timeseries_data')
    if i == 0:

        want_to_import_pv =str(input('Do you need the code to import PV data? (if "n" il will use the '
                                  'corresponding dataset in the TIMESERIES folder:   '))



        if want_to_import_pv == 'y':
            import_pv_data(lat, long, project_life)

        want_to_import_wind = str(input('Do you need the code to import WIND data? (if "n" il will use the '
                                      'corresponding dataset in the TIMESERIES folder:   '))

        if want_to_import_wind == 'y':
            import_wind_data(lat, long, project_life)


    os.chdir(r'../..//..')