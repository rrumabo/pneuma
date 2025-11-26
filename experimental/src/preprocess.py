from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import hashlib

try:
    import xarray as xr
except ImportError as e:
    xr = None

from . import reanalysis as rea  # uses open_dataset + interp_to_grid

def _detect_lon_lat_names(ds) -> tuple[str, str]:
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    lat_name = "latitude"  if "latitude"  in ds.coords else ("lat" if "lat" in ds.coords else None)
    if lon_name is None or lat_name is None:
        raise ValueError(f"Couldn't find lon/lat in coords: {list(ds.coords)}")
    return lon_name, lat_name

def _norm_bbox(bbox):
    """
    Accepts either:
      - dict with keys {lon0, lon1, lat0, lat1}, or
      - a 4-sequence [latN, lonW, latS, lonE] (common GIS export)
    Returns a dict with canonical keys and floats.
    """
    if isinstance(bbox, dict):
        return {
            "lon0": float(bbox["lon0"]),
            "lon1": float(bbox["lon1"]),
            "lat0": float(bbox["lat0"]),
            "lat1": float(bbox["lat1"]),
        }
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        latN, lonW, latS, lonE = map(float, bbox)  # N, W, S, E
        return {"lon0": lonW, "lon1": lonE, "lat0": latS, "lat1": latN}
    raise ValueError("bbox must be dict{lon0,lon1,lat0,lat1} or list [latN,lonW,latS,lonE].")

def _cache_basename(stem: str, Nx: int, Ny: int, bbox, key: str | None) -> str:
    """
    Build a short, stable cache base-name using grid, bbox, and an optional key.
    """
    bb = _norm_bbox(bbox)
    tag = (key or "default").replace("/", "_")
    sig = f"{stem}|{Nx}|{Ny}|{bb['lon0']},{bb['lon1']},{bb['lat0']},{bb['lat1']}|{tag}"
    h8 = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{Nx}x{Ny}_{tag}_{h8}"

def find_cache(nc_path: str | Path, bbox, Nx: int, Ny: int, cache_dir: str | Path, key: str | None) -> str | None:
    """
    Returns the cache .npz path if it exists, otherwise None.
    """
    stem = Path(nc_path).stem
    base = _cache_basename(stem, Nx, Ny, bbox, key)
    p = Path(cache_dir) / f"{base}.npz"
    return str(p) if p.exists() else None

def _robust_subset(ds, bbox: Dict[str, float], lon_name: str, lat_name: str):
    lon_vals = ds[lon_name].values
    lat_vals = ds[lat_name].values
    if lon_vals.size == 0 or lat_vals.size == 0:
        return ds  # nothing to subset

    # handle 0..360 vs -180..180 longitudes
    use_0360 = (np.nanmax(lon_vals) > 180.0)
    conv = (lambda l: l % 360.0) if use_0360 else (lambda l: l)

    lon0 = conv(float(bbox["lon0"]))
    lon1 = conv(float(bbox["lon1"]))
    lat0 = float(bbox["lat0"])
    lat1 = float(bbox["lat1"])

    lon_lo, lon_hi = (lon0, lon1) if lon0 <= lon1 else (lon1, lon0)
    lat_lo, lat_hi = (lat0, lat1) if lat0 <= lat1 else (lat1, lat0)

    lon_asc = bool(lon_vals[0] <= lon_vals[-1])
    lat_asc = bool(lat_vals[0] <= lat_vals[-1])

    lon_slice = slice(lon_lo, lon_hi) if lon_asc else slice(lon_hi, lon_lo)
    lat_slice = slice(lat_lo, lat_hi) if lat_asc else slice(lat_hi, lat_lo)

    ds_sub = ds.sel({lon_name: lon_slice, lat_name: lat_slice})

    # if subsetting produced empty dims, fall back to full file
    sizes = getattr(ds_sub, "sizes", {})
    if int(sizes.get(lat_name, 0)) == 0 or int(sizes.get(lon_name, 0)) == 0:
        return ds
    return ds_sub

def _pick_height_like_var(ds) -> str:
    # Prefer height-like variables; otherwise first data var
    candidates = ["geopotential", "z", "gh", "height"]
    for v in candidates:
        if v in ds.data_vars:
            return v
    # fallback: first variable
    return list(ds.data_vars)[0]

def _make_2d_latlon(da, lon_name: str, lat_name: str):
    # Drop/first-slice any non-spatial dims (e.g., time, level)
    for d in list(da.dims):
        if d not in (lat_name, lon_name):
            da = da.isel({d: 0})
    da = da.squeeze(drop=True).transpose(lat_name, lon_name)
    # rename to the names reanalysis.interp_to_grid expects
    rename_map = {}
    if lon_name != "longitude": rename_map[lon_name] = "longitude"
    if lat_name != "latitude":  rename_map[lat_name]  = "latitude"
    if rename_map:
        da = da.rename(rename_map)
    return da

def preproc_era5_to_npz(
    nc_path: str,
    bbox,
    Nx: int,
    Ny: int,
    use_geos_winds: bool,
    cache_dir: str | Path,
    cache_key: str,
) -> str:
    """
    Open a small ERA5/CMIP file, subset to bbox, interpolate to (Ny,Nx),
    sanitize, and cache to NPZ. Returns the NPZ path.
    """
    if xr is None:
        raise ImportError("xarray is required for preprocess; pip install xarray netCDF4")

    bb = _norm_bbox(bbox)

    # check cache
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(nc_path).stem
    base = _cache_basename(stem, Nx, Ny, bb, cache_key)
    npz_path = cache_dir / f"{base}.npz"
    if npz_path.exists():
        print(f"Using cache: {npz_path}")
        return str(npz_path)

    ds = rea.open_dataset(nc_path)
    lon_name, lat_name = _detect_lon_lat_names(ds)
    ds_sub = _robust_subset(ds, bb, lon_name, lat_name)

    var = _pick_height_like_var(ds_sub)
    da = ds_sub[var]
    if "time" in da.dims:
        da = da.isel(time=0)

    da2d = _make_2d_latlon(da, lon_name, lat_name)

    # Interpolate to model grid
    field = rea.interp_to_grid(da2d, Nx, Ny).astype(np.float64)

    # Sanitize NaN/Inf and normalize scale (keep raw magnitudes; conversion to h happens in run)
    mean_f = float(np.nanmean(field))
    if not np.isfinite(mean_f):
        raise ValueError("Interpolated field is all-NaN or invalid.")
    field = np.nan_to_num(field, nan=mean_f, posinf=mean_f, neginf=mean_f)

    meta = np.array([bb["lon0"], bb["lon1"], bb["lat0"], bb["lat1"]], dtype=float)
    np.savez(npz_path, field=field, meta=meta, var=np.array([var]), stem=np.array([stem]))
    print(f"Cache saved: {npz_path}")
    return str(npz_path)