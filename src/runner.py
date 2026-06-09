from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

from pneuma.core import time_integrators as ti
from pneuma.utils.diagnostic_manager import DiagnosticManager
from pneuma.utils.diagnostics import (
    compute_norms,
    cfl_number_advection,
    cfl_number_diffusion,
    write_metrics_json,
)


@dataclass
class Solution:
    t: np.ndarray           # shape (Ns,)
    y: np.ndarray           # shape (Ns, dim) or (Ns, ...) for PDE states
    meta: Dict[str, Any]    # method, dt, steps, elapsed, user records


def _select_stepper(method: str):
    """
    Return a function with signature: (f, t, y, dt, params) -> y_next
    by adapting integrators that use: step(y, rhs(u,t), t, dt, diagnostics_fn=None).
    """
    name = method.strip().lower()

    def make_adapter(step_fn):
        def _adapt(f, t, y, dt, params):
            # bind params and flip argument order for integrators: rhs(u, t)
            def rhs(u, tt):
                return f(tt, u, params)
            return step_fn(y, rhs, t, dt, None)
        return _adapt

    if name == "euler":
        base = getattr(ti, "euler_step", None)
    elif name in ("rk2", "heun"):
        base = getattr(ti, "rk2_step", None)
    elif name == "rk4":
        base = getattr(ti, "rk4_step", None)
    else:
        raise ValueError("Unknown method. Use: euler, rk2, heun, rk4.")
    if base is None:
        raise ValueError(f"Integrator for '{method}' not found in time_integrators.py")

    return make_adapter(base)

def solve_fixed_step(
    f: Callable[[float, np.ndarray, Dict[str, Any]], np.ndarray],
    *,
    t_span: Tuple[float, float],
    y0: np.ndarray,
    dt: float,
    method: str = "rk4",
    params: Optional[Dict[str, Any]] = None,
    save_every: int = 1,
    callbacks: Optional[Iterable[Callable[[float, np.ndarray, Dict[str, Any]], Optional[bool]]]] = None,
    diagnostics: Optional[DiagnosticManager] = None,
    metrics_out_dir: Optional[Path] = None,
    norm_grid: Optional[Tuple[float, Optional[float]]] = None,
    cfl_specs: Optional[Dict[str, Dict[str, Any]]] = None,
    norm_save_every: int = 1,
) -> Solution:
    """
    Fixed-step time integration for ODEs or semi-discrete PDEs (Method of Lines).
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    t0, t1 = float(t_span[0]), float(t_span[1])
    if t1 <= t0:
        raise ValueError("t_span must satisfy t1 > t0")

    stepper = _select_stepper(method)
    p = params or {}
    cbs = list(callbacks) if callbacks is not None else []

    # Ensure arrays
    y = np.array(y0, dtype=float, copy=True)
    t = t0

    # Diagnostics
    dm = diagnostics or DiagnosticManager()
    dm.reset()
    dm.start()

    # Preallocate conservative upper bound for number of saves
    n_steps_total = int(np.ceil((t1 - t0) / dt))
    n_saves_est = n_steps_total // save_every + 2  # (not strictly needed, kept for clarity)

    t_hist: List[float] = [t]
    y_hist: List[np.ndarray] = [np.array(y, copy=True)]

    # --- Norms and metrics initialization ---
    l2_hist: List[float] = []
    dx_val: Optional[float] = None
    dy_val: Optional[float] = None
    if norm_grid is not None:
        dx_val = float(norm_grid[0])
        dy_val = None if len(norm_grid) == 1 or norm_grid[1] is None else float(norm_grid[1])
        try:
            n0 = compute_norms(y, dx=dx_val, dy=dy_val)
            l2_hist.append(n0["L2"]) 
        except Exception:
            pass

    steps = 0
    while t < t1 - 1e-15:
        # Adjust last dt to land exactly on t1
        dt_eff = min(dt, t1 - t)

        # Advance one step
        y = stepper(f, t, y, dt_eff, p)
        t = t + dt_eff
        steps += 1
        dm.tick()

        # Save history
        if (steps % save_every) == 0 or t >= t1 - 1e-15:
            t_hist.append(t)
            y_hist.append(np.array(y, copy=True))
            if norm_grid is not None and (steps % norm_save_every == 0 or t >= t1 - 1e-15):
                try:
                    nnow = compute_norms(y, dx=dx_val if dx_val is not None else 1.0, dy=dy_val)
                    l2_hist.append(nnow["L2"]) 
                except Exception:
                    pass

        # Callbacks can stop early
        if cbs:
            meta = {"step": steps, "t": t, "dt": dt_eff, "method": method}
            if any(bool(cb(t, y, meta)) for cb in cbs):
                break

    dm.stop()

    # --- Metrics output and CFL computation ---
    cfls: Dict[str, float] = {}
    if cfl_specs:
        for key, spec in cfl_specs.items():
            stype = spec.get("type", "diffusion")
            if stype == "advection":
                cfls[key] = float(
                    cfl_number_advection(
                        dt=float(spec["dt"]),
                        dx=float(spec["dx"]),
                        a=spec.get("a"),
                        u=spec.get("u"),
                    )
                )
            elif stype == "diffusion":
                cfls[key] = float(
                    cfl_number_diffusion(
                        dt=float(spec["dt"]),
                        dx=float(spec["dx"]),
                        nu=float(spec["nu"]),
                        dim=int(spec.get("dim", 1)),
                    )
                )
            elif "value" in spec:
                cfls[key] = float(spec["value"])

    if metrics_out_dir is not None:
        last_state = y_hist[-1]
        if isinstance(last_state, np.ndarray):
            if last_state.ndim == 1:
                grid_info = last_state.shape[0]
            elif last_state.ndim == 2:
                grid_info = list(last_state.shape)
            else:
                grid_info = int(last_state.size)
        else:
            grid_info = 0

        # Let this raise if it's wrong – we *want* to see the bug.
        write_metrics_json(
            out_dir=metrics_out_dir,
            scheme=method,
            grid=grid_info,
            dt=float(dt),
            cfl=cfls,
            norms_over_time={"L2": l2_hist} if l2_hist else {},
            extras={"elapsed_s": float(dm.summary().get("elapsed_s", 0.0))},
        )    

    T = np.asarray(t_hist, dtype=float)
    Y = np.stack(y_hist, axis=0)  # shape (Ns, ...)

    meta = {
        "method": method,
        "dt": float(dt),
        "steps": steps,
        **dm.summary(),
        "norms": {"L2": l2_hist} if l2_hist else {},
        "cfl": cfls,
    }
    return Solution(t=T, y=Y, meta=meta)

def _run_heat1d_from_cfg(cfg):
    """
    Execute a 1D heat equation run from a Config-like object.

    Expects at least:
      - cfg.model == "heat1d"
      - cfg.grid.N (or Nx), cfg.grid.L
      - cfg.time.dt, cfg.time.T
      - cfg.physics.params["alpha"]
      - cfg.ic or cfg.initial_condition with fields:
          type: "gaussian"
          center, sigma, amp
      - cfg.io.outdir
      - optionally cfg.integrator.method
    """
    # --- Extract grid ---
    grid = cfg.grid
    N = int(getattr(grid, "N", getattr(grid, "Nx", 0)))
    if N <= 0:
        raise ValueError("heat1d: grid.N (or Nx) must be positive")

    L = float(getattr(grid, "L", 1.0))
    dx = L / N

    # --- Time config ---
    time_cfg = cfg.time
    dt = float(getattr(time_cfg, "dt", 0.0))
    T = float(getattr(time_cfg, "T", getattr(time_cfg, "t_end", 0.0)))
    if dt <= 0.0 or T <= 0.0:
        raise ValueError("heat1d: dt and T must be positive")

    t_span = (0.0, T)

    # --- Physics (alpha) ---
    phys = getattr(cfg, "physics", None)
    alpha = 0.01
    if phys is not None:
        params_dict = getattr(phys, "params", {}) or {}
        if isinstance(params_dict, dict) and "alpha" in params_dict:
            alpha = float(params_dict["alpha"])

    # --- Initial condition ---
    ic = getattr(cfg, "ic", None)

    # Try alternative attribute name
    if ic is None and hasattr(cfg, "initial_condition"):
        ic = getattr(cfg, "initial_condition")

    # If still missing, fall back to a default Gaussian
    if ic is None:
        print("[pneuma] warning: no IC found in config, using default gaussian IC")
        class _DefaultIC:
            type = "gaussian"
            center = 0.5 * L
            sigma = 0.1 * L
            amp = 1.0
        ic = _DefaultIC()

    ic_type = getattr(ic, "type", "gaussian")
    x0 = float(getattr(ic, "center", 0.5 * L))
    sigma = float(getattr(ic, "sigma", 0.1 * L))
    amp = float(getattr(ic, "amp", 1.0))

    x = np.linspace(0.0, L, N, endpoint=False)

    if ic_type.lower() == "gaussian":
        u0 = amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    else:
        # Fallback: just a small bump
        u0 = amp * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

    # --- Output directory ---
    io_cfg = getattr(cfg, "io", None)
    outdir_str = getattr(io_cfg, "outdir", "outputs/heat1d") if io_cfg is not None else "outputs/heat1d"
    outdir = Path(outdir_str)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Integrator method ---
    integrator_cfg = getattr(cfg, "integrator", None)
    method = getattr(integrator_cfg, "method", "rk4") if integrator_cfg is not None else "rk4"

    # --- Define RHS for heat equation: u_t = alpha * u_xx with periodic BCs ---
    def heat_rhs(t, u, params):
        dx_loc = float(params["dx"])
        alpha_loc = float(params["alpha"])
        # periodic Laplacian
        lap = (np.roll(u, -1) + np.roll(u, 1) - 2.0 * u) / (dx_loc ** 2)
        return alpha_loc * lap

    params = {"dx": dx, "alpha": alpha}

    # --- Run the solver using the generic fixed-step integrator ---
    sol = solve_fixed_step(
        f=heat_rhs,
        t_span=t_span,
        y0=u0,
        dt=dt,
        method=method,
        params=params,
        save_every=1,
        metrics_out_dir=outdir,
        norm_grid=(dx,None)
    )

    u_final = sol.y[-1]

    # --- Save outputs ---
    np.savetxt(outdir / "x.csv", x)
    np.savetxt(outdir / "u_final.csv", u_final)

    # Optional: simple plot
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(x, u0, label="t=0")
        ax.plot(x, u_final, label=f"t={T}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.legend()
        ax.set_title("Heat1D: initial vs final")
        fig.tight_layout()
        fig.savefig(outdir / "u_final.png", dpi=150)
        plt.close(fig)
    except Exception:
        pass

    print(f"[pneuma] heat1d run wrote outputs to {outdir}")
    return sol

def run_from_config(cfg):
    """
    Dispatch runs based on cfg.model.

    v0.2: only heat1d is wired for real.
          Other models fall back to a stub.
    """
    model = getattr(cfg, "model", "unknown")

    if model == "heat1d":
        return _run_heat1d_from_cfg(cfg)

    # Future: add shallow-water, ERA5-driven, etc.
    print(f"[pneuma] (stub) run_from_config for model={model}")
    return None
