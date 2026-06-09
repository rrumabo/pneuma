from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from config import load_config
from src.io.preprocess import preproc_era5_to_npz
from src.models.sw_adapter import rhs, make_initial_sw, sw_cfl
from src.models.sw_adapter import box_smooth
from .runner import solve_fixed_step
from src.utils.forcing import make_forcings
from src.utils.structs import as_dict, find_cache
from src.utils.sanitize import sanitize_ic_and_dt
from dataclasses import is_dataclass, asdict as _dc_asdict
from typing import Any, Mapping, cast

# --- helper: robustly coerce numbers or dicts with numeric keys to float ---
def _as_float(x: Any, *keys, default=np.nan) -> float:
    """Return a float whether x is already a number or a dict with known keys."""
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    if isinstance(x, dict):
        for k in keys:
            if k in x:
                try:
                    return float(x[k])
                except Exception:
                    continue
    return float(default)

def cmd_preproc(cfg):
    if not cfg.data.era5_nc:
        print("No era5_nc in config; nothing to preproc.")
        return
    Path(cfg.cache.dir).mkdir(parents=True, exist_ok=True)
    npz = preproc_era5_to_npz(
        cfg.data.era5_nc, cfg.data.bbox,
        cfg.grid.Nx, cfg.grid.Ny,
        cfg.data.use_geos_winds,
        cfg.cache.dir, cfg.cache.key
    )
    print("Cache saved:", npz)

def cmd_run(cfg):
    """
    Safer two-phase run:
      - Phase A (short spin-up): Calm ERA5-initialized state with stronger damping/viscosity.
      - Phase B (main run): nominal physics with CFL-safe dt.
    Works whether cache NPZ holds (h0,u0,v0) or a single 'field' (e.g., geopotential).
    """
    # grid & physics from config 
    Ny, Nx = int(cfg.grid.Ny), int(cfg.grid.Nx)
    Lx, Ly = float(cfg.grid.Lx), float(cfg.grid.Ly)
    dx, dy = Lx / Nx, Ly / Ny
    print(f"[grid] Nx={Nx} Ny={Ny} Lx={Lx} Ly={Ly} dx={dx:.5g} dy={dy:.5g}")  

    g  = float(cfg.physics.g)
    f  = float(cfg.physics.f)
    nu = float(cfg.physics.nu)
    Du = float(cfg.physics.Du)
    Dv = float(cfg.physics.Dv)

    # base params + minimum floors for stability
    params: dict[str, Any] = {"g": g, "f": f, "nu": nu, "dx": dx, "dy": dy, "Du": Du, "Dv": Dv}
    params["nu"] = max(params.get("nu", 0.0), 2e-4)
    params["Du"] = max(params.get("Du", 0.0), 2e-3)
    params["Dv"] = max(params.get("Dv", 0.0), 2e-3)
    params["scheme"] = "rusanov"
    params.setdefault("hmin", 1e-6)
    params.setdefault("u_cap", 30.0)
    params.setdefault("nu_h", 1e-4)
    # optional forcing
    if getattr(cfg, "forcing", None) is not None:
        Fc = make_forcings(Ny, Nx, as_dict(cfg.forcing))
        params.update(Fc)

    # build IC
    Y0 = None
    npz_path = None
    if getattr(cfg, "cache", None) is not None and cfg.cache.dir and cfg.cache.key:
        npz_path = find_cache(
            getattr(cfg.data, "era5_nc", None),
            getattr(cfg.data, "bbox", None),
            Nx, Ny,
            cfg.cache.dir, cfg.cache.key
        )
        if npz_path is None and getattr(cfg.data, "era5_nc", None):
            Path(cfg.cache.dir).mkdir(parents=True, exist_ok=True)
            npz_path = preproc_era5_to_npz(
                cfg.data.era5_nc, cfg.data.bbox,
                cfg.grid.Nx, cfg.grid.Ny,
                cfg.data.use_geos_winds,
                cfg.cache.dir, cfg.cache.key
            )
            npz_path = Path(npz_path)

        # load cache
        if npz_path is not None and npz_path.exists():
            z = np.load(npz_path, allow_pickle=True)
            if all(k in z.files for k in ("h0", "u0", "v0")):
                h0, u0, v0 = z["h0"], z["u0"], z["v0"]
            else:
                # legacy single-field cache
                field = np.asarray(z["field"], dtype=np.float64)
                varname = str(z["var"][0]) if "var" in z.files else ""
                if varname.lower() in {"geopotential", "z", "gh"}:
                    h0 = np.maximum(field / g, 1e-6)  # convert to meters
                else:
                    m = float(np.nanmean(field)) if np.isfinite(np.nanmean(field)) else 1.0
                    if m == 0.0: m = 1.0
                    h0 = np.maximum((field / m) * 1.0, 1e-6)
                u0 = np.zeros_like(h0); v0 = np.zeros_like(h0)
            # sanitize and calm before stacking (prevents overflows)
            h0 = np.nan_to_num(h0, nan=1.0, posinf=1.0, neginf=1.0)
            h0 = np.clip(h0, 1e-6, 10.0)
            u0 = np.nan_to_num(u0, nan=0.0, posinf=0.0, neginf=0.0)
            v0 = np.nan_to_num(v0, nan=0.0, posinf=0.0, neginf=0.0)
            u0.fill(0.0); v0.fill(0.0)  # calm start
            Y0 = np.stack([h0, u0, v0])

    # fallback if no cache/NetCDF
    if Y0 is None:
        Ytmp = make_initial_sw(Ny, Nx, h0=1.0, jet_amp=0.08)
        h0, u0, v0 = Ytmp[0].copy(), Ytmp[1].copy(), Ytmp[2].copy()
        # sanitize & calm
        h0 = np.nan_to_num(h0, nan=1.0, posinf=1.0, neginf=1.0)
        h0 = np.clip(h0, 1e-6, 10.0)
        u0 = np.zeros_like(h0); v0 = np.zeros_like(h0)
        Y0 = np.stack([h0, u0, v0])

    # time config (dt may be None if user provided CFL)
    dt_req_raw = getattr(cfg.time, "dt", None)
    dt_req = None if dt_req_raw is None else float(dt_req_raw)
    T_main     = float(cfg.time.T)
    method     = str(cfg.time.method)
    save_every = int(cfg.time.save_every)
    out_dir    = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # choose CFL target from config if provided
    cfl_main = float(getattr(cfg.time, "cfl", 0.15) or 0.15)

    # sanitize IC and pick a CFL-safe dt (caps winds if user-requested dt is too large)
    Y0_s, dt_used_main, spinup_info = sanitize_ic_and_dt(
        Y0, g, dx, dy, dt_req, cfl_target=cfl_main
    )
    # compute a CFL-based dt for reporting (or fallback)
    dxmin = min(dx, dy)
    c0 = np.sqrt(np.maximum(g * np.maximum(Y0_s[0], 1e-6), 0.0))
    vmax0 = float(np.nanmax(np.abs(Y0_s[1]))) + float(np.nanmax(c0))
    dt_cfl_main = float(cfl_main * dxmin / max(vmax0, 1e-12))

    # Fallback if sanitize didn’t provide a valid CFL dt
    if not np.isfinite(dt_cfl_main) or dt_cfl_main <= 0:
        dxmin = min(dx, dy)
        c0 = np.sqrt(np.maximum(g * np.maximum(Y0_s[0], 1e-6), 0.0))
        vmax0 = float(np.nanmax(np.abs(Y0_s[1]))) + float(np.nanmax(c0))
        dt_cfl_main = cfl_main * dxmin / max(vmax0, 1e-12)

    # If dt_used_main is still invalid, use the CFL dt
    if not np.isfinite(dt_used_main) or dt_used_main <= 0:
        dt_used_main = dt_cfl_main

    # Guard against mis-specified domain lengths producing absurd dt
    if dt_used_main > 1e-2:
        print(f"WARN: dt_used_main={dt_used_main:.3g} looks too big; "
              f"check Lx/Ly and Nx/Ny. Clamping to dt_cfl_main.")
        dt_used_main = dt_cfl_main

    # Ensure we don’t end up with only start/end snapshots:
    # If save_every is invalid or too large relative to the estimated step count,
    # reduce it to target ~100 snapshots (but at least 1).
    nsteps_est = int(np.ceil(T_main / max(dt_used_main, 1e-12)))
    if save_every <= 0 or save_every >= nsteps_est:
        save_every = max(1, nsteps_est // 100)

    # Phase A: short spin-up — stronger dissipation
    # relax them
    spin_params = dict(params)
    spin_params["nu"] = max(spin_params.get("nu", 0.0), 4e-4)
    spin_params["Du"] = max(spin_params.get("Du", 0.0), 4e-3)
    spin_params["Dv"] = max(spin_params.get("Dv", 0.0), 4e-3)

    #small
    dxmin = min(dx, dy)
    c0 = np.sqrt(np.maximum(g * np.maximum(Y0_s[0], 1e-6), 0.0))
    vmax0 = float(np.nanmax(np.abs(Y0_s[1]))) + float(np.nanmax(c0))
    cfl_spin = 0.6 * cfl_main
    dt_cfl_spin = cfl_spin * dxmin / max(vmax0, 1e-12)
    dt_spin = min(dt_used_main, dt_cfl_spin)
    if not np.isfinite(dt_spin) or dt_spin <= 0:
        dt_spin = max(1e-6, 0.5 * max(dt_used_main, 1e-6))
    T_spin  = min(0.02, 0.1 * T_main)  #smalll
    se_spin = max(1, int(np.ceil(T_spin / max(dt_spin, 1e-12))))
    print(f"[spinup] dt={dt_spin:.3g}  T={T_spin:.3g}  steps≈{se_spin}", flush=True)

    solA = solve_fixed_step(
        f=rhs,
        t_span=(0.0, T_spin),
        y0=Y0_s,
        dt=dt_spin,
        method=method,
        params=spin_params,
        save_every=se_spin,
        metrics_out_dir=None,   #no artifacts
        norm_grid=(dx, dy),
        cfl_specs={"gw": {"type": "advection", "dt": dt_spin, "dx": dxmin,
                          "u": np.abs(Y0_s[1]) + np.sqrt(np.maximum(g * np.maximum(Y0_s[0], 1e-6), 0.0))}},
    )
    Y_spun = solA.y[-1] if len(solA.y) else Y0_s

    # --- re-sanitize & calm after spin-up, then CAP winds, THEN pick dt ---
    Y_spun = np.nan_to_num(Y_spun, nan=0.0, posinf=0.0, neginf=0.0)
    Y_spun[0] = np.clip(Y_spun[0], 1e-6, 10.0)   # keep h in a sane range
    Y_spun[0] = box_smooth(Y_spun[0], iters=1)

    # Cap winds BEFORE choosing dt
    u_cap = float(params.get("u_cap", 5.0))  # m/s cap; tweak via config if needed
    umax0 = float(np.nanmax(np.sqrt(Y_spun[1] * Y_spun[1] + Y_spun[2] * Y_spun[2])))
    if np.isfinite(umax0) and umax0 > u_cap:
        scale = u_cap / umax0
        Y_spun[1] *= scale
        Y_spun[2] *= scale

    # Re-sanitize (keeps positivity etc.), but we will recompute dt from CFL on the capped state
    Y_spun, _dt_sanitize, _info2 = sanitize_ic_and_dt(
        Y_spun, g, dx, dy, dt_req, cfl_target=cfl_main
    )

    # CFL dt on the *capped* state
    dxmin = min(dx, dy)
    c0_spun = np.sqrt(np.maximum(g * np.maximum(Y_spun[0], 1e-6), 0.0))
    u_max   = float(np.nanmax(np.abs(Y_spun[1])))
    c_max   = float(np.nanmax(c0_spun))
    dt_cfl_main = float(cfl_main * dxmin / max(u_max + c_max, 1e-12))

    # Choose dt (respect user dt if provided, otherwise CFL)
    dt_used_main = float(_dt_sanitize) if (dt_req is not None) else float(dt_cfl_main)
    if not np.isfinite(dt_used_main) or dt_used_main <= 0:
        dt_used_main = dt_cfl_main

    # Guard against absurdly large dt from bad domain params
    if dt_used_main > 1e-2:
        print(f"WARN: dt_used_main={dt_used_main:.3g} looks too big; "
              f"clamping to CFL {dt_cfl_main:.3g}.")
        dt_used_main = dt_cfl_main

    # Update save_every after final dt is known
    nsteps_main = int(np.ceil(T_main / max(dt_used_main, 1e-12)))
    if save_every <= 0 or save_every >= nsteps_main:
        save_every = max(1, nsteps_main // 100)

    print(f"[postcap] u_max={u_max:.3g}  c_max={c_max:.3g}  "
          f"dt_cfl={dt_cfl_main:.3g}  dt_used={dt_used_main:.3g}", flush=True)
    sol = solve_fixed_step(
        f=rhs,
        t_span=(0.0, T_main),
        y0=Y_spun,
        dt=dt_used_main,
        method=method,
        params=params,
        save_every=save_every,
        metrics_out_dir=out_dir,
        norm_grid=(dx, dy),
        cfl_specs={"gw": {"type": "advection", "dt": dt_used_main, "dx": dxmin,
                          "u": np.abs(Y_spun[1]) + np.sqrt(np.maximum(g * np.maximum(Y_spun[0], 1e-6), 0.0))}},
    )

    # summary
    cfls = [sw_cfl(Y, params, dt_used_main) for Y in sol.y]
    print({
        "snapshots": len(sol.t),
        "cfl_max": float(np.nanmax(cfls)),
        "dt_requested": (None if dt_req is None else float(dt_req)),
        "dt_cfl_main": float(dt_cfl_main),
        "dt_used_main": float(dt_used_main),
        "out_dir": str(out_dir),
        "cache_used": str(npz_path) if npz_path and Path(npz_path).exists() else None,
        "spinup": {"T": float(T_spin), "dt": float(dt_spin)},
        "u_cap": float(params.get("u_cap", 5.0))
    })

def cmd_validate(cfg):
    # lightweight check: shapes & dt sanity
    assert int(cfg.grid.Nx) > 0 and int(cfg.grid.Ny) > 0
    # allow either fixed dt or CFL
    dt_val  = getattr(cfg.time, "dt", None)
    cfl_val = getattr(cfg.time, "cfl", None)
    dt_ok  = (dt_val is not None) and float(dt_val) > 0
    cfl_ok = (cfl_val is not None) and float(cfl_val) > 0
    assert (dt_ok or cfl_ok) and float(cfg.time.T) > 0
    print("Config looks sane.")

def main():
    ap = argparse.ArgumentParser("meteo-utils CLI")
    ap.add_argument("--config","-c", required=True, help="YAML config path")
    ap.add_argument("command", choices=["preproc","run","validate"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.command=="preproc": cmd_preproc(cfg)
    elif args.command=="run": cmd_run(cfg)
    else: cmd_validate(cfg)

if __name__=="__main__":
    main()