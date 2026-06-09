import pandas as pd
import os

DATA_DIR = "data"  # base folder for actuals/forecasts
YEARS = range(2015, 2026)

def load_actuals(year):
    load_file = os.path.join(DATA_DIR, "actual", f"DE_load_actuals_{year}.csv")
    solar_file = os.path.join(DATA_DIR, "actual", f"DE_solar_actuals_{year}.csv")
    wind_onshore_file = os.path.join(DATA_DIR, "actual", f"DE_wind_onshore_actuals_{year}.csv")
    wind_offshore_file = os.path.join(DATA_DIR, "actual", f"DE_wind_offshore_actuals_{year}.csv")

    df_load = pd.read_csv(load_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')
    df_solar = pd.read_csv(solar_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')
    df_wind_onshore = pd.read_csv(wind_onshore_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')
    df_wind_offshore = pd.read_csv(wind_offshore_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')

    # Convert indices to timezone-aware Europe/Berlin if not already
    for df in [df_load, df_solar, df_wind_onshore, df_wind_offshore]:
        # ensure a DatetimeIndex
        df.index = pd.to_datetime(df.index, utc=True)
        tz = getattr(df.index, 'tz', None)
        if tz is None:
            df.index = df.index.tz_convert('Europe/Berlin')
        else:
            df.index = df.index.tz_convert('Europe/Berlin')

    # Resample all dataframes to hourly resolution
    df_load = df_load.resample('1h').mean()
    df_solar = df_solar.resample('1h').mean()
    df_wind_onshore = df_wind_onshore.resample('1h').mean()
    df_wind_offshore = df_wind_offshore.resample('1h').mean()

    df_actuals = df_load.join(df_solar).join(df_wind_onshore).join(df_wind_offshore)
    return df_actuals

def load_forecasts(year):
    load_file = os.path.join(DATA_DIR, "forecasts", f"DE_load_forecast_{year}.csv")
    solar_file = os.path.join(DATA_DIR, "forecasts", f"DE_solar_forecast_{year}.csv")
    wind_onshore_file = os.path.join(DATA_DIR, "forecasts", f"DE_wind_onshore_forecast_{year}.csv")
    wind_offshore_file = os.path.join(DATA_DIR, "forecasts", f"DE_wind_offshore_forecast_{year}.csv")

    df_load = pd.read_csv(load_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')
    df_solar = pd.read_csv(solar_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')
    df_wind_onshore = pd.read_csv(wind_onshore_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')
    df_wind_offshore = pd.read_csv(wind_offshore_file, parse_dates=['utc_timestamp']).set_index('utc_timestamp')

    df_load = df_load.iloc[:, 0].to_frame(name='DE_load_forecast')
    df_solar = df_solar.iloc[:, 0].to_frame(name='DE_solar_forecast')
    df_wind_onshore = df_wind_onshore.iloc[:, 0].to_frame(name='DE_wind_onshore_forecast')
    df_wind_offshore = df_wind_offshore.iloc[:, 0].to_frame(name='DE_wind_offshore_forecast')

    # Convert indices to timezone-aware Europe/Berlin if not already
    for df in [df_load, df_solar, df_wind_onshore, df_wind_offshore]:
        # ensure a DatetimeIndex
        df.index = pd.to_datetime(df.index, utc=True)
        tz = getattr(df.index, 'tz', None)
        if tz is None:
            df.index = df.index.tz_convert('Europe/Berlin')
        else:
            df.index = df.index.tz_convert('Europe/Berlin')

    # Resample all dataframes to hourly resolution
    df_load = df_load.resample('1h').mean()
    df_solar = df_solar.resample('1h').mean()
    df_wind_onshore = df_wind_onshore.resample('1h').mean()
    df_wind_offshore = df_wind_offshore.resample('1h').mean()

    df_forecasts = df_load.join(df_solar).join(df_wind_onshore).join(df_wind_offshore)
    return df_forecasts

def merge_actuals_forecasts():
    for year in YEARS:
        df_actuals = load_actuals(year)
        df_forecasts = load_forecasts(year)

        # Alignment ensures indices are equal; no strict check needed

        df_merged = df_actuals.join(df_forecasts)

        df_merged['solar_error'] = (
            df_merged['DE_solar_forecast'] - df_merged['DE_solar_actual']
        )
        df_merged['wind_onshore_error'] = (
            df_merged['DE_wind_onshore_forecast'] - df_merged['DE_wind_onshore_actual']
        )
        df_merged['wind_offshore_error'] = (
            df_merged['DE_wind_offshore_forecast'] - df_merged['DE_wind_offshore_actual']
        )
        df_merged['load_error'] = (
            df_merged['DE_load_forecast'] - df_merged['DE_load_actual']
        )

        output_dir = os.path.join(DATA_DIR, "processed")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"DE_merged_{year}.csv")
        df_merged.to_csv(output_file)

if __name__ == "__main__":
    merge_actuals_forecasts()