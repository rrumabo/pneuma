import os
import pandas as pd
from entsoe.entsoe import EntsoePandasClient

# === CONFIG ===
api_key = '75950ee9-4888-446f-9c45-cf89c47e69d6'
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_folder = os.path.join(project_dir, 'data', 'forecasts')
os.makedirs(data_folder, exist_ok=True)

client = EntsoePandasClient(api_key=api_key)
years = range(2015, 2026)
country_code = 'DE'

for year in years:
    start = pd.Timestamp(f'{year}-01-01T00:00', tz='Europe/Berlin')
    end = pd.Timestamp(f'{year}-12-31T23:00', tz='Europe/Berlin')

    # --- Renewable forecasts ---
    df_res = client.query_wind_and_solar_forecast(
        country_code=country_code,
        start=start,
        end=end
    )

    df_solar = df_res[['Solar']].copy()
    df_solar.columns = ['DE_solar_forecast']
    df_solar.to_csv(
        os.path.join(data_folder, f'DE_solar_forecast_{year}.csv'),
        index_label='utc_timestamp'
    )

    df_wind_offshore = df_res[['Wind Offshore']].copy()
    df_wind_offshore.columns = ['DE_wind_offshore_forecast']
    df_wind_offshore.to_csv(
        os.path.join(data_folder, f'DE_wind_offshore_forecast_{year}.csv'),
        index_label='utc_timestamp'
    )

    df_wind_onshore = df_res[['Wind Onshore']].copy()
    df_wind_onshore.columns = ['DE_wind_onshore_forecast']
    df_wind_onshore.to_csv(
        os.path.join(data_folder, f'DE_wind_onshore_forecast_{year}.csv'),
        index_label='utc_timestamp'
    )

    # --- Load forecast ---
    df_load = client.query_load_forecast(
        country_code=country_code,
        start=start,
        end=end
    )

    if isinstance(df_load, pd.Series):
        df_load = df_load.to_frame(name='DE_load_forecast')
    else:
        df_load = df_load.iloc[:, 0].to_frame(name='DE_load_forecast')

    df_load.to_csv(
        os.path.join(data_folder, f'DE_load_forecast_{year}.csv'),
        index_label='utc_timestamp'
    )