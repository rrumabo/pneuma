from pathlib import Path
import numpy as np
import xarray as xr
from typing import Dict, Tuple, Optional, Union, Any, Mapping
import datetime as dt

def open_era5(path: str) -> xr.Dataset:
    """
    Open an ERA5 file (GRIB or NetCDF) with the correct xarray engine.
    - GRIB: engine="cfgrib"
    - NetCDF: engine="netcdf4" (fallback to default if not available)
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"[Metis] ERA5 file not found: {p}")

    suffix = p.suffix.lower()
    if suffix in {".grib", ".grb"} or p.name.lower().endswith(".grib2"):
        ds = xr.open_dataset(p, engine="cfgrib")
    elif suffix in {".nc", ".nc4", ".cdf"}:
        # Let xarray choose netCDF engine if netcdf4 not installed
        try:
            ds = xr.open_dataset(p, engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(p)
    else:
        # Try cfgrib first, then default
        try:
            ds = xr.open_dataset(p, engine="cfgrib")
        except Exception:
            ds = xr.open_dataset(p)
    return ds

def slice_time(ds: xr.Dataset, t_index: int = 0, t_value: Optional[Union[str, "np.datetime64"]] = None) -> xr.Dataset:
    """
    Return a single-time slice.
    - If t_value is provided (e.g., "2020-07-01T03:00"), select/nearest that time.
    - Else, use integer index (isel).
    """
    if "time" not in ds.coords:
        return ds
    if t_value is not None:
        return ds.sel(time=np.datetime64(t_value), method="nearest")
    return ds.isel(time=t_index)

def interp_time(ds: xr.Dataset, when: Union[str, "np.datetime64"]) -> xr.Dataset:
    """
    Linearly interpolate the dataset to an arbitrary timestamp between hourly steps.
    """
    if "time" not in ds.coords:
        return ds
    t = np.datetime64(when)
    return ds.interp(time=t) 

def interp_to_grid(ds: xr.Dataset, ny: int, nx: int, Ly: float, Lx: float) -> dict:
    """
    Regrid a single-time ERA5 slice to a uniform ny×nx model grid in index space.

    Accepts either (lat, lon) or (y, x) coords in the input dataset.
    Produces fields on model index coordinates y∈[0,1], x∈[0,1] (model uses Ly/Lx separately).

    NOTE: We deliberately uniformize to [0,1] to avoid CRS/units headaches here.
    """
    # Detect input coordinate names
    if "y" in ds.coords and "x" in ds.coords:
        y_name, x_name = "y", "x"
    else:
        y_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
        x_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
        if y_name is None or x_name is None:
            raise ValueError("interp_to_grid: dataset must have either (y,x) or (latitude/longitude) coordinates")

    dsn = ds.rename({y_name: "y", x_name: "x"})

    # Ensure monotonic ascending for interpolation
    if dsn.indexes.get("y", None) is not None and getattr(dsn.indexes["y"], "is_monotonic_decreasing", False):
        dsn = dsn.sortby("y", ascending=True)
    if dsn.indexes.get("x", None) is not None and getattr(dsn.indexes["x"], "is_monotonic_decreasing", False):
        dsn = dsn.sortby("x", ascending=True)

    # Normalize source coords to [0,1] and build uniform target
    dsn = dsn.assign_coords(
        y=np.linspace(0.0, 1.0, dsn.sizes["y"]),
        x=np.linspace(0.0, 1.0, dsn.sizes["x"]),
    )
    target = dsn.interp(
        y=np.linspace(0.0, 1.0, ny),
        x=np.linspace(0.0, 1.0, nx),
        kwargs={"fill_value": "extrapolate"},
    )

    # Return plain float32 arrays for known names
    out: Dict[str, np.ndarray] = {}
    for name in ("t2m", "u10", "v10", "skt"):
        if name in target:
            arr = target[name].values
            out[name] = np.asarray(arr, dtype=np.float32)
    return out

def make_initial_from_era5(fields: Dict[str, np.ndarray], h0: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (T_init[K], h_init[m], u_init[m/s], v_init[m/s]) mapped to solver variables.

    Notes:
    - ERA5 t2m is Kelvin. Keep Kelvin internally for physics; convert to Celsius at output time.
    - ERA5 u10/v10 are m/s at 10 m height. We pass them through unchanged here.
    """
    ny, nx = next(iter(fields.values())).shape

    # Temperature (Kelvin)
    T = np.asarray(fields.get("t2m", np.full((ny, nx), 300.0, dtype=np.float32)), dtype=np.float32)

    # Shallow-water depth and velocities
    h = np.full((ny, nx), h0, dtype=np.float32)
    u = np.zeros_like(h, dtype=np.float32)
    v = np.zeros_like(h, dtype=np.float32)

    if "u10" in fields:
        u = np.asarray(fields["u10"], dtype=np.float32)  # m/s at 10 m
    if "v10" in fields:
        v = np.asarray(fields["v10"], dtype=np.float32)  # m/s at 10 m

    return T, h, u, v

def kelvin_to_celsius(T_K: np.ndarray) -> np.ndarray:
    return np.asarray(T_K, dtype=np.float32) - np.float32(273.15)

def _npz_save(file: str | Path, **arrays: Any) -> None:
    # Pylance-safe wrapper around numpy.savez_compressed
    np.savez_compressed(file=str(file), **arrays)

def _npz_load(file: str | Path) -> dict[str, np.ndarray]:
    data = np.load(str(file), allow_pickle=False)
    return {k: np.asarray(data[k]) for k in data.files}

def to_cache(fields: Mapping[str, np.ndarray], path_npz: Union[str, Path]) -> None:
    p = Path(path_npz).expanduser().resolve()
    arrays: dict[str, Any] = {k: np.asarray(v) for k, v in fields.items()}
    _npz_save(p, **arrays)

def from_cache(path_npz: Union[str, Path]) -> Dict[str, np.ndarray]:
    p = Path(path_npz).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"[Metis] cache not found: {p}")
    return _npz_load(p)

def mean_bias_correction(model: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Return model - mean_bias(model, ref)."""
    bias = float(np.nanmean(model - ref))
    return model - bias