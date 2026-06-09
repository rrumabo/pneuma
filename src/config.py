from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml, re
from typing import Optional

class ConfigError(ValueError):
    """User-facing config errors with clear messages."""
    pass


def validate_heat1d_dict(d: dict) -> None:
    """Basic schema validation for the heat1d YAML."""
    required_sections = ["grid", "time", "physics", "initial_condition", "io"]
    for sec in required_sections:
        if sec not in d:
            raise ConfigError(f"Missing required section '{sec}' in config.")

    grid = d["grid"]
    if not ("N" in grid or ("Nx" in grid and "Ny" in grid)):
        raise ConfigError("grid must contain either (N, L) or (Nx, Ny, Lx, Ly) for heat1d.")

    time = d["time"]
    if "dt" not in time or "T" not in time:
        raise ConfigError("time must contain 'dt' and 'T' for heat1d.")

    phys = d["physics"]
    if phys.get("pde") not in (None, "heat"):
        raise ConfigError("physics.pde must be 'heat' (or omitted) for heat1d configs.")
    params = phys.get("params", {})
    if "alpha" not in params:
        raise ConfigError("physics.params.alpha is required for heat1d.")

    ic = d["initial_condition"]
    if ic.get("type") != "gaussian":
        raise ConfigError(
            f"initial_condition.type must be 'gaussian' for heat1d, got {ic.get('type')!r}"
        )
    for k in ("center", "sigma", "amp"):
        if k not in ic:
            raise ConfigError(f"initial_condition.{k} is required for heat1d.")

    io = d["io"]
    if "outdir" not in io:
        raise ConfigError("io.outdir is required.")

@dataclass
class GridCfg:
    Nx:int; Ny:int; Lx:float; Ly:float

@dataclass
class TimeCfg:
    dt: Optional[float] = None
    T: float = 0.1
    method: str = "rk4"
    save_every: int = 20
    # New: allow specifying a CFL target in YAML; CLI can compute dt from this
    cfl: Optional[float] = None

@dataclass
class PhysCfg:
    g:float; f:float; nu:float=0.0; Du:float=0.0; Dv:float=0.0

@dataclass
class ForcingCfg:
    Fh: object=None  # None | float | str "sin(amp=...,period=...)"

@dataclass
class DataCfg:
    era5_nc: str|None=None
    bbox: dict|None=None
    use_geos_winds: bool=False

@dataclass
class CacheCfg:
    dir:str="cache"; key:str="default"

@dataclass
class OutCfg:
    dir:str="outputs/v5_run"; keep_frames:bool=False

@dataclass
class RunCfg:
    model:str
    grid:GridCfg
    time:TimeCfg
    physics:PhysCfg
    forcing:ForcingCfg=field(default_factory=ForcingCfg)
    data:DataCfg=field(default_factory=DataCfg)
    cache:CacheCfg=field(default_factory=CacheCfg)
    output:OutCfg=field(default_factory=OutCfg)

def _coerce_forcing(s):
    if s is None or isinstance(s,(int,float)): return s
    m=re.fullmatch(r"sin\(amp=([\deE\.\-+]+),\s*period=([\deE\.\-+]+)\)", str(s))
    if m: return ("sin", float(m.group(1)), float(m.group(2)))
    raise ValueError(f"Unsupported forcing spec: {s}")

def load_config(path: str | Path) -> RunCfg:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}

        # decide model name, default to heat1d
        model_name = str(d.get("model", "heat1d")).lower()

        if model_name == "heat1d":
            validate_heat1d_dict(d)
        else:
            # For v1.0, we only support heat1d through this loader
            raise ConfigError(f"Unsupported model '{model_name}' in this version.")

    # --- physics block + shims ---
    phys_d = dict(d.get("physics", {}) or {})

    # allow 'drag' shortcut → Du,Dv (only if not already set)
    drag = phys_d.pop("drag", None)
    if drag is not None:
        drag = float(drag)
        phys_d.setdefault("Du", drag)
        phys_d.setdefault("Dv", drag)

    # --- forcing block + migration of misplaced keys ---
    forcing_d = dict(d.get("forcing", {}) or {})

    # migrate mistakenly placed physics keys from forcing → physics
    for k in ("Du", "Dv", "nu", "f", "g"):
        if k in forcing_d and k not in phys_d:
            phys_d[k] = float(forcing_d.pop(k))

    # coerce Fh (None | number | "sin(amp=...,period=...)")
    Fh = _coerce_forcing(forcing_d.get("Fh"))
    forcing_d = {} if Fh is None else {"Fh": Fh}

    # --- time/output shims ---
    time_d = dict(d.get("time", {}) or {})
    out_d  = dict(d.get("output", {}) or {})

    # if someone put output.save_every in YAML, move it to time.save_every
    if "save_every" in out_d and "save_every" not in time_d:
        time_d["save_every"] = int(out_d.pop("save_every"))

    # --- data shims (bbox can be list/tuple or dict) ---
    data_d = dict(d.get("data", {}) or {})
    bbox = data_d.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        data_d["bbox"] = {
            "lon0": float(bbox[0]), "lon1": float(bbox[1]),
            "lat0": float(bbox[2]), "lat1": float(bbox[3]),
        }
    elif isinstance(bbox, dict):
        for k in ("lon0", "lon1", "lat0", "lat1"):
            if k in bbox:
                bbox[k] = float(bbox[k])
        data_d["bbox"] = bbox
    elif bbox is None:
        pass
    else:
        raise TypeError("data.bbox must be [lon0, lon1, lat0, lat1] or a dict with those keys.")

    # --- grid shims: support both 1D (N, L) and 2D (Nx, Ny, Lx, Ly) ---
    raw_grid = dict(d.get("grid", {}) or {})
    if "Nx" in raw_grid and "Ny" in raw_grid:
        # 2D case: assume user provided full info
        grid_d = raw_grid
    elif "N" in raw_grid and "L" in raw_grid:
        # 1D case: map to a degenerate 2D grid with Ny=1, Ly=1.0
        N = int(raw_grid["N"])
        L = float(raw_grid["L"])
        grid_d = {
            "Nx": N,
            "Ny": 1,
            "Lx": L,
            "Ly": 1.0,
        }
    else:
        raise KeyError("grid section must contain either (Nx, Ny, Lx, Ly) or (N, L)")

    # --- path resolution & default cache key ---

    # expand user/home and make absolute paths
    if data_d.get("era5_nc"):
        data_d["era5_nc"] = str(Path(data_d["era5_nc"]).expanduser().resolve())

    cache_d = dict(d.get("cache", {}) or {})
    cache_d["dir"] = str(Path(cache_d.get("dir", "cache")).expanduser().resolve())
    out_d["dir"]   = str(Path(out_d.get("dir", "outputs/v5_run")).expanduser().resolve())

    # auto-build a cache key if none provided
    if not cache_d.get("key"):
        base = Path(data_d["era5_nc"]).stem if data_d.get("era5_nc") else "era5_default"
        cache_d["key"] = f"{base}_{int(grid_d['Nx'])}x{int(grid_d['Ny'])}"

    # write back sanitized sections
    d["physics"] = phys_d
    d["forcing"] = forcing_d
    d["time"]    = time_d
    d["output"]  = out_d
    d["data"]    = data_d
    d["cache"]   = cache_d
    d["grid"]    = grid_d

    # --- model-specific physics payload ----------------------------------------
    # model_name was computed earlier (defaulting to 'heat1d')

    if model_name == "heat1d":
        # For heat1d, PhysCfg isn't actually used by the solver.
        # We just give it a harmless dummy config.
        phys_cfg = PhysCfg(g=0.0, f=0.0, nu=0.0, Du=0.0, Dv=0.0)
    else:
        # For shallow-water-style models, pull only the keys PhysCfg expects.
        g  = float(phys_d.get("g", 9.81))
        f  = float(phys_d.get("f", 0.0))
        nu = float(phys_d.get("nu", 0.0))
        Du = float(phys_d.get("Du", 0.0))
        Dv = float(phys_d.get("Dv", 0.0))
        phys_cfg = PhysCfg(g=g, f=f, nu=nu, Du=Du, Dv=Dv)

    # --- build dataclasses ------------------------------------------------------
    return RunCfg(
        model=model_name,
        grid=GridCfg(**d["grid"]),
        time=TimeCfg(**time_d),
        physics=phys_cfg,
        forcing=ForcingCfg(**forcing_d),
        data=DataCfg(**data_d),
        cache=CacheCfg(**cache_d),
        output=OutCfg(**out_d),
    )