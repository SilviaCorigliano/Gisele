import requests
import pandas as pd

##
# PV
##

'''lat = -16.5941
long = 36.5955
project_life = 25'''


def import_pv_data(lat, long, project_life):

    token = '556d9ea27f35f2e26ac9ce1552a3f702e35a8596  '
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()
    # Send token header with each request
    s.headers = {'Authorization': 'Token ' + token}

    data = []

    time_step = list(range(2001, 2019))
    if project_life > len(time_step):
        extra_years = project_life - len(time_step)
        extra_time_step = time_step[:extra_years]
        time_step = list(time_step) + list(extra_time_step)

    else:
        time_step = time_step[:project_life]

    url = api_base + 'data/pv'

    print(time_step)
    for years in time_step:
        print(years)
        args = {
            'lat': lat,
            'lon': long,
            'date_from': str(years) + '-01-01',
            'date_to': str(years) + '-12-31',
            'dataset': 'merra2',
            'capacity': 1.0,
            'system_loss': 0,
            'tracking': 0,
            'tilt': 35,
            'azim': 180,
            'format': 'json',
            'metadata': False,
            'raw': True
        }

        r = s.get(url, params=args)

        # Parse JSON to get a pandas.DataFrame
        df = pd.read_json(r.text, orient='index')
        data.append(df)

    df_final = pd.concat(data)

    print(df_final)
    # pv_ref = pd.read_csv('pv.csv')
    df_final.to_csv('pv.csv')

    print("Solar Data imported")


def import_wind_data(lat, long, project_life):

    token = '40ef709a6becd58c69de1f09cb2d5b61ceb6686e '
    api_base = 'https://www.renewables.ninja/api/'

    s = requests.session()
    # Send token header with each request
    s.headers = {'Authorization': 'Token ' + token}

    data = []

    time_step = list(range(2001, 2019))

    if project_life > len(time_step):
        extra_years = project_life - len(time_step)
        extra_time_step = time_step[:extra_years]
        time_step = list(time_step) + list(extra_time_step)

    else:
        time_step = time_step[:project_life]

    url = api_base + 'data/wind'

    print(time_step)
    for years in time_step:
        print(years)
        args = {
            'lat': lat,
            'lon': long,
            'date_from': str(years) + '-01-01',
            'date_to': str(years) + '-12-31',
            'dataset': 'merra2',
            'capacity': 1.0,
            'height': 100,
            'turbine': 'Vestas V80 2000',
            'format': 'json',
            'metadata': False,
            'raw': True
        }

        r = s.get(url, params=args)

        # Parse JSON to get a pandas.DataFrame
        df = pd.read_json(r.text, orient='index')
        data.append(df)

    df_final = pd.concat(data)

    print(df_final)
    df_final.to_csv('wind.csv')

    print("Wind Data imported")

# import_pv_data(lat, long, project_life)
# import_wind_data(lat, long, project_life)
