import os
import pandas as pd
from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

api_key = '75950ee9-4888-446f-9c45-cf89c47e69d6'
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
actual_folder = os.path.join(project_dir, 'data', 'actual')
os.makedirs(actual_folder, exist_ok=True)
client = EntsoePandasClient(api_key=api_key)
country_code = 'DE'

for year in range(2015, 2026):
    start = pd.Timestamp(f'{year}-01-01T00:00', tz='Europe/Berlin')
    end = pd.Timestamp(f'{year}-12-31T23:00', tz='Europe/Berlin')

    df_solar = client.query_generation(
        country_code=country_code,
        start=start,
        end=end,
        psr_type='B16'
    )

    # Handle MultiIndex structure (Solar → Actual Aggregated)
    if isinstance(df_solar.columns, pd.MultiIndex):
        df_solar = df_solar[('Solar', 'Actual Aggregated')]

    # Ensure single clean column
    if isinstance(df_solar, pd.Series):
        df_solar = df_solar.to_frame(name='DE_solar_actual')
    else:
        df_solar = pd.DataFrame({'DE_solar_actual': df_solar.squeeze()})
    df_solar.index.name = 'utc_timestamp'
    solar_file = os.path.join(actual_folder, f'DE_solar_actuals_{year}.csv')
    df_solar.to_csv(solar_file)

    # --- Fetch Wind Actual (Onshore + Offshore) separately ---
    try:
        df_wind_on = client.query_generation(
            country_code=country_code,
            start=start,
            end=end,
            psr_type='B19'  # Onshore (updated)
        )
    except NoMatchingDataError:
        df_wind_on = None

    try:
        df_wind_off = client.query_generation(
            country_code=country_code,
            start=start,
            end=end,
            psr_type='B18'  # Offshore (updated)
        )
    except NoMatchingDataError:
        df_wind_off = None

    # Prepare onshore wind DataFrame
    if df_wind_on is None:
        df_wind_on = pd.DataFrame(index=df_solar.index)
        df_wind_on['DE_wind_onshore_actual'] = pd.NA
    else:
        if isinstance(df_wind_on, pd.Series):
            df_wind_on = df_wind_on.to_frame()
        df_wind_on = df_wind_on.iloc[:, 0].to_frame(name='DE_wind_onshore_actual')
    df_wind_on.index.name = 'utc_timestamp'
    wind_on_file = os.path.join(actual_folder, f'DE_wind_onshore_actuals_{year}.csv')
    df_wind_on.to_csv(wind_on_file)

    # Prepare offshore wind DataFrame
    if df_wind_off is None:
        df_wind_off = pd.DataFrame(index=df_solar.index)
        df_wind_off['DE_wind_offshore_actual'] = pd.NA
    else:
        if isinstance(df_wind_off, pd.Series):
            df_wind_off = df_wind_off.to_frame()
        df_wind_off = df_wind_off.iloc[:, 0].to_frame(name='DE_wind_offshore_actual')
    df_wind_off.index.name = 'utc_timestamp'
    wind_off_file = os.path.join(actual_folder, f'DE_wind_offshore_actuals_{year}.csv')
    df_wind_off.to_csv(wind_off_file)

    df_load = client.query_load(country_code=country_code, start=start, end=end)
    if isinstance(df_load, pd.Series):
        df_load = df_load.to_frame()
    df_load = df_load.iloc[:, 0].to_frame(name='DE_load_actual')
    df_load.index.name = 'utc_timestamp'
    load_file = os.path.join(actual_folder, f'DE_load_actuals_{year}.csv')
    df_load.to_csv(load_file)