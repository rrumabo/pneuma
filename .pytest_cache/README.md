# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

# Meteo Utils — Toy PDE & Urban Microclimate Sandbox

Small, testable code for shallow-water PDEs and a minimal Urban Heat Island (UHI) layer, with an optional ERA5 path. Goal: build a reliable backbone for city-scale digital-twin experiments.

## Quick start (no data)
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/uhi_toy.ipynb

Outputs → outputs/uhi_toy/ (T_final.png, metrics.json).

Optional ERA5
pip install xarray netCDF4 cdsapi
python scripts/get_era5_cyprus.py   # writes data/era5_cyprus_2020-07-01.nc
jupyter lab notebooks/era5_to_sw.ipynb

What works (v5)
	•	PDE: 1D/2D heat, Burgers, 1-layer shallow-water (periodic), RK2/RK4.
	•	UHI toy: 4-field [h,u,v,T] with simple surface-energy source + drag.
	•	Data: ERA5 stub + tiny Cyprus slice; robust subsetting + normalization.
	•	Diagnostics: CFL guardrails, mass conservation, metrics.json.

Repo: 
src/ (core, models, io, utils)  notebooks/  scripts/  tests/
data/ (gitignored)              outputs/ (gitignored)

----------------------------------------------------------------------------------------------------------------------------------------------------------------

### Pneuma (formerly *Meteo Utils*) — Toy PDE & Urban Microclimate Sandbox

Small, testable code for shallow‑water PDEs and a minimal Urban Heat Island (UHI) layer, with an optional ERA5 path. Goal: a reliable backbone for city‑scale digital‑twin experiments.

---

## What’s new in v5
- **CLI + YAML configs**: `validate`, `preproc`, `run`.
- **ERA5 preprocess & caching**: robust subsetting → cached `.npz` for fast runs.
- **Auto‑Δt from CFL** with short **spin‑up** and optional **velocity cap**.
- **Upwind (Rusanov) option** in the SW core (more robust than central; still simple).
- New utilities: **sanitize ICs**, **struct helpers**, and **boundary helpers**.
- Example config: `examples/cyprus_sw.yaml`. Tests: `tests/test_v5_config_cli.py`.

> Repo folder renamed locally to **pneuma**. The GitHub repo may still show the old name *meteo‑utils*; paths in the code use `src/...`, so the rename is safe.

---

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
# (Optional for ERA5)
pip install xarray netCDF4 cdsapi
```

---

## Quick starts

### A) No data (toy run)
Open the UHI toy notebook or run a SW toy case (see `notebooks/` and `scripts/`).

### B) With config + CLI (recommended)
1) **Validate config**
```bash
python -m src.cli -c examples/cyprus_sw.yaml validate
```
2) **Preprocess ERA5** (creates a cache; falls back to a tiny synthetic field if no NetCDF is found)
```bash
python -m src.cli -c examples/cyprus_sw.yaml preproc
```
3) **Run** (uses cache, picks a stable `dt` if `time.dt: null`)
```bash
python -m src.cli -c examples/cyprus_sw.yaml run
```
Outputs go to `outputs/...` and include `final_state.npz`, quick images (if enabled), and `metrics.json`.

---

## Example config (`examples/cyprus_sw.yaml`)
```yaml
model: shallow_water

grid: { Nx: 192, Ny: 144, Lx: 1.0, Ly: 0.75 }

# If dt is null, CLI computes a stable dt from CFL after a short spin-up
time:
  T: 0.2
  dt: null
  spinup: { T: 0.02, dt: 4.7e-5 }

physics:
  g: 9.81
  f: 1.0e-4
  nu: 1.0e-4      # small viscosity
  Du: 2.0e-3      # linear drag (u)
  Dv: 2.0e-3      # linear drag (v)
  u_cap: 30.0     # cap winds during spin-up (m/s)
  adv_fac: 1.0    # 0..1 to scale advection if needed

forcing:
  Fh: null        # domain heating; can be scalar or 2D array (optional)

data:
  era5_nc: data/era5_cyprus_2020-07-01.nc  # optional; preproc falls back if absent
  bbox: [30.0, 36.0, 33.0, 36.5]           # lon0, lon1, lat0, lat1
  use_geos_winds: false

cache: { dir: cache, key: default }

output: { dir: outputs/v5_cyprus, save_every: 20 }
```

---

## CLI commands
- `validate` — sanity‑check grid/time values and dt/CFL consistency.
- `preproc`  — read ERA5 (if present) → subset (handles 0..360 vs −180..180) → interpolate to model grid → save `.npz` cache.
- `run`      — calm/sanitize ICs → optional spin‑up → compute safe `dt` if needed → integrate and write artifacts.

Printed run summary includes grid info, spin‑up settings, `dt` used, and CFL estimates.

---

## Repo layout
```
src/
  cli.py                     # CLI entry (validate / preproc / run)
  config.py                  # YAML → typed config, defaults & shims
  core/time_integrators.py   # RK steppers
  io/
    reanalysis.py            # mock + xarray stubs for ERA5/CMIP
    preprocess.py            # ERA5 → cache .npz (subset + interp)
  models/
    shallow_water.py         # SW core (central or Rusanov), CFL helpers
    sw_adapter.py            # Routes 3‑field vs 4‑field states
    boundaries.py            # Utility BC helpers
  utils/
    sanitize.py              # Calm/clip ICs, pick dt from CFL
    structs.py               # lightweight struct helpers
notebooks/
  era5_to_sw.ipynb           # ERA5 slice → SW init
  uhi_toy.ipynb              # UHI minimal toy
scripts/
  get_era5_cyprus.py         # tiny ERA5 sample downloader
  run_sw_rusanov.py          # example script (optional)
  smoke_rusanov.py           # tiny smoke test (optional)
examples/
  cyprus_sw.yaml             # example run config
tests/
  test_v4_validation.py
  test_v5_config_cli.py
cache/    (gitignored)
data/     (gitignored)
outputs/  (gitignored)
```

---

## Troubleshooting
- **Blow‑ups / NaNs**: use `time.dt: null` (auto‑dt), reduce `adv_fac`, increase `nu` or `Du/Dv`, keep `u_cap` ≲ 30, and prefer the Rusanov flux. Smaller grids are also easier.
- **ERA5 not found / bbox empty**: `preproc` falls back to a small synthetic field; check your `bbox` and file lon convention.
- **Imports from scripts**: run commands from the repo root so `src/` is on `PYTHONPATH`.

---

## License
TBD.

## Credit
If you build on this, please link back to this repo. Proper citation will be added at v1.0.