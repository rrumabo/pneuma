import numpy as np
from pathlib import Path
from typing import Optional

from src.runner import solve_fixed_step
from src.models.sw_adapter import rhs, sw_cfl, make_initial_sw
from src.utils.forcing import make_forcings

def _safe_dt(params, Y0):
    g, dx, dy = params["g"], params["dx"], params["dy"]
    c0 = float(np.sqrt(g * float(np.mean(Y0[0]))))
    dx_min = min(dx, dy)
    return 0.25 * dx_min / max(c0, 1e-12)  # conservative

def test_mass_conservation_no_forcing(tmp_path: Optional[Path] = None):
    Ny, Nx = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / Nx, Ly / Ny
    params = {"g": 9.81, "f": 1e-4, "nu": 1e-4, "dx": dx, "dy": dy, "Du": 2e-3, "Dv": 2e-3}

    Y0 = make_initial_sw(Ny, Nx, h0=1.0, jet_amp=0.05)
    Fcfg = make_forcings(Ny, Nx, {"Fh": None, "Du": params["Du"], "Dv": params["Dv"]})
    p = {**params, **Fcfg}

    dt = _safe_dt(p, Y0)
    T  = 0.2

    sol = solve_fixed_step(
        f=rhs, t_span=(0.0, T), y0=Y0, dt=dt, method="rk4",
        params=p, save_every=20, metrics_out_dir=(tmp_path or Path("outputs")/"_test_mass"),
        norm_grid=(dx, dy),
        cfl_specs={"gw": {"type":"advection","dt":dt,"dx":min(dx,dy),"u":np.abs(Y0[1])+np.sqrt(p["g"]*Y0[0])}},
    )

    mean_h = np.array([float(np.mean(Y[0])) for Y in sol.y])
    drift = float(mean_h[-1] - mean_h[0])

    # tolerances: very small; diffusion/drag shouldn’t change domain sum in periodic
    assert np.isfinite(mean_h).all()
    assert abs(drift) < 5e-4, f"Mass drift too large: {drift}"

    # also check CFL stayed under a reasonable cap
    cfls = [sw_cfl(Y, p, dt) for Y in sol.y]
    assert float(np.nanmax(cfls)) < 0.35, f"CFL too high: {np.nanmax(cfls)}"

def test_constant_Fh_mean_drift(tmp_path: Optional[Path] = None):
    Ny, Nx = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / Nx, Ly / Ny
    params = {"g": 9.81, "f": 1e-4, "nu": 1e-4, "dx": dx, "dy": dy, "Du": 2e-3, "Dv": 2e-3}

    Y0 = make_initial_sw(Ny, Nx, h0=1.0, jet_amp=0.0)  # simpler baseline
    Fh_const = 1e-3
    Fcfg = make_forcings(Ny, Nx, {"Fh": Fh_const, "Du": params["Du"], "Dv": params["Dv"]})
    p = {**params, **Fcfg}

    dt = _safe_dt(p, Y0)
    T  = 0.2

    sol = solve_fixed_step(
        f=rhs, t_span=(0.0, T), y0=Y0, dt=dt, method="rk4",
        params=p, save_every=20, metrics_out_dir=(tmp_path or Path("outputs")/"_test_forcing"),
        norm_grid=(dx, dy),
        cfl_specs={"gw": {"type":"advection","dt":dt,"dx":min(dx,dy),"u":np.abs(Y0[1])+np.sqrt(p["g"]*Y0[0])}},
    )

    mh0 = float(np.mean(sol.y[0][0]))
    mhT = float(np.mean(sol.y[-1][0]))
    delta_meas = mhT - mh0
    delta_exp  = Fh_const * T

    # within ~20% is fine for this coarse/time-discrete check
    rel_err = abs(delta_meas - delta_exp) / max(abs(delta_exp), 1e-12)
    assert rel_err < 0.2, f"Mean(h) drift off: meas={delta_meas}, exp={delta_exp}, rel_err={rel_err:.2%}"

    # keep CFL in check
    cfls = [sw_cfl(Y, p, dt) for Y in sol.y]
    assert float(np.nanmax(cfls)) < 0.35