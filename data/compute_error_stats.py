import os
from typing import Dict, List

import numpy as np
import pandas as pd

DATA_DIR = "data/processed"
YEARS = range(2015, 2026)

ERROR_MAP: Dict[str, str] = {
    "solar": "solar_error",
    "wind_onshore": "wind_onshore_error",
    "wind_offshore": "wind_offshore_error",
    "load": "load_error",
}


def compute_metric_row(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()

    if s.empty:
        return {
            "n": 0,
            "mean_error": np.nan,
            "std_error": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "min_error": np.nan,
            "max_error": np.nan,
            "p50": np.nan,
            "p75": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }

    return {
        "n": int(s.shape[0]),
        "mean_error": float(s.mean()),
        "std_error": float(s.std(ddof=1)),
        "mae": float(s.abs().mean()),
        "rmse": float(np.sqrt((s**2).mean())),
        "min_error": float(s.min()),
        "max_error": float(s.max()),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
    }


def load_merged_year(year: int) -> pd.DataFrame:
    file_path = os.path.join(DATA_DIR, f"DE_merged_{year}.csv")
    df = pd.read_csv(file_path, parse_dates=["utc_timestamp"])
    df["utc_timestamp"] = pd.to_datetime(df["utc_timestamp"], utc=True).dt.tz_convert("Europe/Berlin")
    df["hour"] = df["utc_timestamp"].dt.hour
    df["month"] = df["utc_timestamp"].dt.month
    return df


def build_global_rows(df: pd.DataFrame, year: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for variable, error_col in ERROR_MAP.items():
        row = {
            "year": year,
            "variable": variable,
            **compute_metric_row(df[error_col]),
        }
        rows.append(row)
    return rows


def build_hourly_rows(df: pd.DataFrame, year: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for variable, error_col in ERROR_MAP.items():
        for hour, group in df.groupby("hour", sort=True):
            row = {
                "year": year,
                "variable": variable,
                "hour": int(hour),
                **compute_metric_row(group[error_col]),
            }
            rows.append(row)
    return rows


def build_monthly_rows(df: pd.DataFrame, year: int) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for variable, error_col in ERROR_MAP.items():
        for month, group in df.groupby("month", sort=True):
            row = {
                "year": year,
                "variable": variable,
                "month": int(month),
                **compute_metric_row(group[error_col]),
            }
            rows.append(row)
    return rows


def compute_error_stats() -> None:
    global_rows: List[Dict[str, float]] = []
    hourly_rows: List[Dict[str, float]] = []
    monthly_rows: List[Dict[str, float]] = []

    for year in YEARS:
        file_path = os.path.join(DATA_DIR, f"DE_merged_{year}.csv")
        if not os.path.exists(file_path):
            print(f"Skipping {year}: missing merged file")
            continue

        df = load_merged_year(year)
        global_rows.extend(build_global_rows(df, year))
        hourly_rows.extend(build_hourly_rows(df, year))
        monthly_rows.extend(build_monthly_rows(df, year))
        print(f"Processed {year}")

    global_df = pd.DataFrame(global_rows)
    hourly_df = pd.DataFrame(hourly_rows)
    monthly_df = pd.DataFrame(monthly_rows)

    global_df.to_csv(os.path.join(DATA_DIR, "error_stats_global.csv"), index=False)
    hourly_df.to_csv(os.path.join(DATA_DIR, "error_stats_hourly.csv"), index=False)
    monthly_df.to_csv(os.path.join(DATA_DIR, "error_stats_monthly.csv"), index=False)

    print("Saved:")
    print(os.path.join(DATA_DIR, "error_stats_global.csv"))
    print(os.path.join(DATA_DIR, "error_stats_hourly.csv"))
    print(os.path.join(DATA_DIR, "error_stats_monthly.csv"))


if __name__ == "__main__":
    compute_error_stats()