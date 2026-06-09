from __future__ import annotations
import numpy as np
from src.pdes.shallow_water import rhs_sw_2d, cfl_gravity

def rhs(t: float, Y: np.ndarray, params: dict) -> np.ndarray:
    """
    Adapter so the rest of the code can call a single RHS:
    - If Y has 3 fields [h,u,v], use the shallow-water RHS directly.
    - If Y has 4 fields [h,u,v,T], evolve [h,u,v] and keep T frozen (dT/dt = 0).
      (You can later plug a thermal tendency here.)
    """
    if Y.shape[0] == 3:
        return rhs_sw_2d(t, Y, params)

    if Y.shape[0] == 4:
        d3 = rhs_sw_2d(t, Y[:3], params)
        dT = np.zeros_like(Y[0])
        return np.stack([d3[0], d3[1], d3[2], dT])

    raise ValueError(f"Unsupported state size: {Y.shape}")

def sw_cfl(Y: np.ndarray, params: dict, dt: float) -> float:
    """CFL helper that works for 3- or 4-field states."""
    Y3 = Y[:3] if Y.shape[0] == 4 else Y
    g  = float(params["g"])
    dx = float(params["dx"])
    dy = float(params["dy"])
    return float(cfl_gravity(u=Y3[1], v=Y3[2], h=Y3[0], g=g, dt=dt, dx=dx, dy=dy))

def make_initial_sw(Ny: int, Nx: int, h0: float = 1.0, jet_amp: float = 0.08) -> np.ndarray:
    """
    Simple initial condition: uniform depth, sinusoidal zonal jet in u, zero v.
    Returns a 3-field state [h,u,v].
    """
    Y0 = np.zeros((3, Ny, Nx), dtype=float)
    Y0[0].fill(h0)
    yy = np.linspace(0.0, 1.0, Ny, endpoint=False)
    u_profile = jet_amp * np.sin(2.0 * np.pi * yy)   # smooth jet in y
    Y0[1] = u_profile[:, None]                       # same across x
    # Y0[2] already zeros
    return Y0