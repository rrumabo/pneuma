import numpy as np
from typing import Dict, Tuple, Optional

# utilities for finite differences
def _ddx(f: np.ndarray, dx: float) -> np.ndarray:
    """Central difference in x (axis=1) with periodic BC."""
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)


def _ddy(f: np.ndarray, dy: float) -> np.ndarray:
    """Central difference in y (axis=0) with periodic BC."""
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)


def _laplacian(f: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """2D Laplacian with periodic BC."""
    return (
        (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dx * dx)
        + (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dy * dy)
    )


# CFL helper (helps with validation])
def cfl_gravity(u: np.ndarray, v: np.ndarray, h: np.ndarray,
                g: float, dt: float, dx: float, dy: float) -> float:
    """Compute CFL number based on gravity-wave speed and velocity."""
    c = np.sqrt(np.maximum(g * h, 0.0))
    sx = np.nanmax(np.abs(u) + c)
    sy = np.nanmax(np.abs(v) + c)
    return float(max(sx * dt / dx, sy * dt / dy))


# Rusanov fluxes
def _rusanov_flux_x(h, hu, hv, g: float, hmin: float, u_cap: float):
    """Rusanov flux at i+1/2 in x direction (periodic)."""
    hL = h
    hR = np.roll(h, -1, axis=1)
    huL = hu
    huR = np.roll(hu, -1, axis=1)
    hvL = hv
    hvR = np.roll(hv, -1, axis=1)

    hLs = np.maximum(hL, hmin)
    hRs = np.maximum(hR, hmin)

    uL = huL / hLs
    vL = hvL / hLs
    uR = huR / hRs
    vR = hvR / hRs

    # physical flux F(U) in x
    FL_h  = huL
    FL_hu = huL * uL + 0.5 * g * hL * hL
    FL_hv = huL * vL

    FR_h  = huR
    FR_hu = huR * uR + 0.5 * g * hR * hR
    FR_hv = huR * vR

    # max wave speed
    sL = np.abs(uL) + np.sqrt(g * hLs)
    sR = np.abs(uR) + np.sqrt(g * hRs)
    smax = np.maximum(sL, sR)
    smax = np.nan_to_num(
        smax,
        nan=0.0,
        posinf=u_cap + np.sqrt(g * 10.0),
        neginf=0.0,
    )
    smax = np.clip(smax, 0.0, u_cap + np.sqrt(g * 10.0))

    # Rusanov flux
    Fh  = 0.5 * (FL_h  + FR_h)  - 0.5 * smax * (hR  - hL)
    Fhu = 0.5 * (FL_hu + FR_hu) - 0.5 * smax * (huR - huL)
    Fhv = 0.5 * (FL_hv + FR_hv) - 0.5 * smax * (hvR - hvL)
    return Fh, Fhu, Fhv


def _rusanov_flux_y(h, hu, hv, g: float, hmin: float, u_cap: float):
    """Rusanov flux at j+1/2 in y direction (periodic)."""
    hL = h
    hR = np.roll(h, -1, axis=0)
    huL = hu
    huR = np.roll(hu, -1, axis=0)
    hvL = hv
    hvR = np.roll(hv, -1, axis=0)

    hLs = np.maximum(hL, hmin)
    hRs = np.maximum(hR, hmin)

    uL = huL / hLs
    vL = hvL / hLs
    uR = huR / hRs
    vR = hvR / hRs

    # physical flux G(U) in y
    GL_h  = hvL
    GL_hu = huL * vL
    GL_hv = hvL * vL + 0.5 * g * hL * hL

    GR_h  = hvR
    GR_hu = huR * vR
    GR_hv = hvR * vR + 0.5 * g * hR * hR

    sL = np.abs(vL) + np.sqrt(g * hLs)
    sR = np.abs(vR) + np.sqrt(g * hRs)
    smax = np.maximum(sL, sR)
    smax = np.nan_to_num(
        smax,
        nan=0.0,
        posinf=u_cap + np.sqrt(g * 10.0),
        neginf=0.0,
    )
    smax = np.clip(smax, 0.0, u_cap + np.sqrt(g * 10.0))

    Gh  = 0.5 * (GL_h  + GR_h)  - 0.5 * smax * (hR  - hL)
    Ghu = 0.5 * (GL_hu + GR_hu) - 0.5 * smax * (huR - huL)
    Ghv = 0.5 * (GL_hv + GR_hv) - 0.5 * smax * (hvR - hvL)
    return Gh, Ghu, Ghv


# RHS: shallow-water
def rhs_sw_2d(t: float, Y: np.ndarray, p: Dict) -> np.ndarray:
    """
    Rusanov/Godunov shallow-water RHS on periodic grid.

    Y: (3, Ny, Nx) = [h, u, v]         (velocities form)
    p: dict with keys:
       "dx","dy","g","f","nu","Du","Dv","Fh","hmin","u_cap" (some optional)
    """
    h, u, v = Y[0], Y[1], Y[2]

# sanitize state to avoid NaN/Inf & extreme values
    hmin = float(p.get("hmin", 1e-4))
    u_cap = float(p.get("u_cap", 30.0))
    h = np.nan_to_num(h, nan=1.0, posinf=1.0, neginf=1.0)
    u = np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    h = np.clip(h, hmin, 10.0)
    u = np.clip(u, -u_cap, u_cap)
    v = np.clip(v, -u_cap, u_cap)

    g    = float(p.get("g", 9.81))
    f    = float(p.get("f", 0.0))
    dx   = float(p["dx"]); dy = float(p["dy"])
    nu   = float(p.get("nu", 0.0))
    Du   = float(p.get("Du", 0.0))
    Dv   = float(p.get("Dv", 0.0))
    Fh_p = p.get("Fh", 0.0)

# Conservative variables
    h_safe = np.maximum(h, hmin)
    hu = h * u
    hv = h * v

# Numerical fluxes
    Fh, Fhu, Fhv = _rusanov_flux_x(h, hu, hv, g, hmin, u_cap)
    Gh, Ghu, Ghv = _rusanov_flux_y(h, hu, hv, g, hmin, u_cap)

# Divergence of fluxes -> conservative tendencies
    div_x_h  = (Fh  - np.roll(Fh,  1, axis=1)) / dx
    div_x_hu = (Fhu - np.roll(Fhu, 1, axis=1)) / dx
    div_x_hv = (Fhv - np.roll(Fhv, 1, axis=1)) / dx

    div_y_h  = (Gh  - np.roll(Gh,  1, axis=0)) / dy
    div_y_hu = (Ghu - np.roll(Ghu, 1, axis=0)) / dy
    div_y_hv = (Ghv - np.roll(Ghv, 1, axis=0)) / dy

    dhdt_cons   = -(div_x_h  + div_y_h)
    dhu_dt_cons = -(div_x_hu + div_y_hu)
    dhv_dt_cons = -(div_x_hv + div_y_hv)

# Sources: Coriolis, viscosity, linear drag
    cor_hu = -f * h * v
    cor_hv =  f * h * u

    visc_hu = nu * _laplacian(hu, dx, dy) if nu else 0.0
    visc_hv = nu * _laplacian(hv, dx, dy) if nu else 0.0

    drag_hu = -Du * hu if Du else 0.0
    drag_hv = -Dv * hv if Dv else 0.0

    dhu_dt = dhu_dt_cons + cor_hu + visc_hu + drag_hu
    dhv_dt = dhv_dt_cons + cor_hv + visc_hv + drag_hv

# External mass forcing Fh (scalar, array, or None)
    if isinstance(Fh_p, np.ndarray):
        dhdt = dhdt_cons + Fh_p
    elif Fh_p is None:
        dhdt = dhdt_cons
    else:
        dhdt = dhdt_cons + float(Fh_p)

# Convert conservative momentum tendencies -> velocity tendencies
    h_safe = np.maximum(h, hmin)
    dudt = (dhu_dt - u * dhdt) / h_safe
    dvdt = (dhv_dt - v * dhdt) / h_safe

    dhdt = np.nan_to_num(dhdt)
    dudt = np.nan_to_num(dudt)
    dvdt = np.nan_to_num(dvdt)

    return np.stack([dhdt, dudt, dvdt], axis=0)


# time stepping + demo
def forward_euler(rhs, Y0: np.ndarray, params: Dict,
                  dt: float, steps: int) -> np.ndarray:
    """Simple forward-Euler integrator for Y_t = rhs(t, Y, params)."""
    Y = Y0.copy()
    t = 0.0
    for _ in range(steps):
        dY = rhs(t, Y, params)
        Y = Y + dt * dY
        t += dt
    return Y


def make_initial_sw_state(nx: int, ny: int,
                        h0: float = 1.0,
                        bump_amp: float = 0.05) -> np.ndarray:
    """
    Simple initial condition:
    - base depth h0
    - 2D Gaussian bump in the center
    - zero initial velocities
    """
    y = np.linspace(-1.0, 1.0, ny)
    x = np.linspace(-1.0, 1.0, nx)
    X, Y = np.meshgrid(x, y)
    bump = np.exp(-(X**2 + Y**2) / 0.1)

    h = h0 + bump_amp * bump
    u = np.zeros_like(h)
    v = np.zeros_like(h)

    Y0 = np.stack([h, u, v], axis=0)
    return Y0


def run_shallow_water(nx: int, ny: int,
                dx: float, dy: float,
                dt: float, t_end: float,
                params: Dict,
                Y0: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Run a shallow-water solver and return final (h, u, v) fields.

    Parameters in `params`:
        g, f, nu, Du, Dv, h0, bump_amp, hmin, u_cap, Fh, ...
    Y0: optional initial state array (3, ny, nx). If None, uses default initial condition.
    """
    p = dict(params)  # copy
    p["dx"] = dx
    p["dy"] = dy

    if Y0 is None:
        h0 = float(p.get("h0", 1.0))
        bump_amp = float(p.get("bump_amp", 0.05))
        Y0 = make_initial_sw_state(nx, ny, h0=h0, bump_amp=bump_amp)

    steps = int(t_end / dt)
    Y_final = forward_euler(rhs_sw_2d, Y0, p, dt, steps)
    h, u, v = Y_final[0], Y_final[1], Y_final[2]
    return np.stack([h, u, v], axis=0)