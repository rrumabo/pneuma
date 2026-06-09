import numpy as _np
from typing import Dict as _Dict

# ----------------------- utilities (periodic) -----------------------
def _ddx(f: _np.ndarray, dx: float) -> _np.ndarray:
    # central difference, periodic, x = axis 1
    return (_np.roll(f, -1, axis=1) - _np.roll(f, 1, axis=1)) / (2.0 * dx)

def _ddy(f: _np.ndarray, dy: float) -> _np.ndarray:
    # central difference, periodic, y = axis 0
    return (_np.roll(f, -1, axis=0) - _np.roll(f, 1, axis=0)) / (2.0 * dy)

def _laplacian(f: _np.ndarray, dx: float, dy: float) -> _np.ndarray:
    return ((_np.roll(f, -1, axis=1) - 2.0 * f + _np.roll(f, 1, axis=1)) / (dx * dx) +
            (_np.roll(f, -1, axis=0) - 2.0 * f + _np.roll(f, 1, axis=0)) / (dy * dy))

# CFL helper (kept for sw_adapter/tests)
def cfl_gravity(u: _np.ndarray, v: _np.ndarray, h: _np.ndarray,
                g: float, dt: float, dx: float, dy: float) -> float:
    c = _np.sqrt(_np.maximum(g * h, 0.0))
    sx = _np.nanmax(_np.abs(u) + c)
    sy = _np.nanmax(_np.abs(v) + c)
    return float(max(sx * dt / dx, sy * dt / dy))

# ----------------------- Rusanov fluxes ----------------------------
def _rusanov_flux_x(h, hu, hv, g: float, hmin: float, u_cap: float):
    """
    Compute numerical flux across i+1/2 in x direction using Rusanov.
    Inputs are *cell-centered* arrays; neighbor is roll(-1, axis=1).
    Returns 3 arrays: Fh_hat, Fhu_hat, Fhv_hat at the i+1/2 interfaces.
    """
    # Left (L) = cell i, Right (R) = cell i+1
    hL = h
    hR = _np.roll(h, -1, axis=1)
    huL = hu
    huR = _np.roll(hu, -1, axis=1)
    hvL = hv
    hvR = _np.roll(hv, -1, axis=1)

    hLs = _np.maximum(hL, hmin)
    hRs = _np.maximum(hR, hmin)

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

    # wave speed (max eigenvalue)
    sL = _np.abs(uL) + _np.sqrt(g * hLs)
    sR = _np.abs(uR) + _np.sqrt(g * hRs)
    smax = _np.maximum(sL, sR)
    smax = _np.nan_to_num(smax, nan=0.0, posinf=u_cap + _np.sqrt(g * 10.0), neginf=0.0)
    smax = _np.clip(smax, 0.0, u_cap + _np.sqrt(g * 10.0))

    # Rusanov flux 0.5*(F_L+F_R) - 0.5*smax*(U_R-U_L)
    Fh = 0.5 * (FL_h + FR_h) - 0.5 * smax * (hR - hL)
    Fhu = 0.5 * (FL_hu + FR_hu) - 0.5 * smax * (huR - huL)
    Fhv = 0.5 * (FL_hv + FR_hv) - 0.5 * smax * (hvR - hvL)
    return Fh, Fhu, Fhv

def _rusanov_flux_y(h, hu, hv, g: float, hmin: float, u_cap: float):
    """
    Compute numerical flux across j+1/2 in y direction using Rusanov.
    Neighbor is roll(-1, axis=0).
    """
    hL = h
    hR = _np.roll(h, -1, axis=0)
    huL = hu
    huR = _np.roll(hu, -1, axis=0)
    hvL = hv
    hvR = _np.roll(hv, -1, axis=0)

    hLs = _np.maximum(hL, hmin)
    hRs = _np.maximum(hR, hmin)

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

    sL = _np.abs(vL) + _np.sqrt(g * hLs)
    sR = _np.abs(vR) + _np.sqrt(g * hRs)
    smax = _np.maximum(sL, sR)
    smax = _np.nan_to_num(smax, nan=0.0, posinf=u_cap + _np.sqrt(g * 10.0), neginf=0.0)
    smax = _np.clip(smax, 0.0, u_cap + _np.sqrt(g * 10.0))

    Gh = 0.5 * (GL_h + GR_h) - 0.5 * smax * (hR - hL)
    Ghu = 0.5 * (GL_hu + GR_hu) - 0.5 * smax * (huR - huL)
    Ghv = 0.5 * (GL_hv + GR_hv) - 0.5 * smax * (hvR - hvL)
    return Gh, Ghu, Ghv

# ----------------------- RHS: Rusanov (default) --------------------
def rhs_sw_2d_rusanov(t: float, Y: _np.ndarray, p: _Dict) -> _np.ndarray:
    """
    Rusanov/Godunov shallow-water RHS on periodic grid.
    Y: (3, Ny, Nx) = [h, u, v]  (velocities)
    p: {"g","f","dx","dy","nu","Du","Dv","Fh"(opt),"hmin"(opt)}
    Returns dY/dt with same shape.
    """
    h, u, v = Y[0], Y[1], Y[2]

    # --- sanitize state to avoid NaN/Inf & extreme values ---
    hmin = float(p.get("hmin", 1e-4))
    u_cap = float(p.get("u_cap", 30.0))
    h = _np.nan_to_num(h, nan=1.0, posinf=1.0, neginf=1.0)
    u = _np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    v = _np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    h = _np.clip(h, hmin, 10.0)
    u = _np.clip(u, -u_cap, u_cap)
    v = _np.clip(v, -u_cap, u_cap)

    g    = float(p.get("g", 9.81))
    f    = float(p.get("f", 0.0))
    dx   = float(p["dx"]); dy = float(p["dy"])
    nu   = float(p.get("nu", 0.0))
    Du   = float(p.get("Du", 0.0))
    Dv   = float(p.get("Dv", 0.0))
    Fh_p = p.get("Fh", 0.0)

    # Conservative variables
    h_safe = _np.maximum(h, hmin)
    hu = h * u
    hv = h * v

    # Numerical fluxes
    Fh, Fhu, Fhv = _rusanov_flux_x(h, hu, hv, g, hmin, u_cap)
    Gh, Ghu, Ghv = _rusanov_flux_y(h, hu, hv, g, hmin, u_cap)

    # Divergence of fluxes -> conservative tendencies
    div_x_h  = (Fh  - _np.roll(Fh,  1, axis=1)) / dx
    div_x_hu = (Fhu - _np.roll(Fhu, 1, axis=1)) / dx
    div_x_hv = (Fhv - _np.roll(Fhv, 1, axis=1)) / dx

    div_y_h  = (Gh  - _np.roll(Gh,  1, axis=0)) / dy
    div_y_hu = (Ghu - _np.roll(Ghu, 1, axis=0)) / dy
    div_y_hv = (Ghv - _np.roll(Ghv, 1, axis=0)) / dy

    dhdt_cons  = -(div_x_h  + div_y_h)
    dhu_dt_cons = -(div_x_hu + div_y_hu)
    dhv_dt_cons = -(div_x_hv + div_y_hv)

    # Sources: Coriolis, viscosity (on momenta), linear drag, optional Fh
    # Coriolis in momentum form: (hu)_t += -f * h * v ; (hv)_t += +f * h * u
    cor_hu = -f * h * v
    cor_hv =  f * h * u

    visc_hu = nu * _laplacian(hu, dx, dy) if nu else 0.0
    visc_hv = nu * _laplacian(hv, dx, dy) if nu else 0.0

    drag_hu = -Du * hu if Du else 0.0
    drag_hv = -Dv * hv if Dv else 0.0

    # Add sources to conservative momentum tendencies
    dhu_dt = dhu_dt_cons + cor_hu + visc_hu + drag_hu
    dhv_dt = dhv_dt_cons + cor_hv + visc_hv + drag_hv

    # External mass forcing Fh (scalar or 2D)
    if isinstance(Fh_p, _np.ndarray):
        dhdt = dhdt_cons + Fh_p
    elif Fh_p is None:
        dhdt = dhdt_cons
    else:
        dhdt = dhdt_cons + float(Fh_p)

    # Convert conservative (h,hu,hv) tendencies to (h,u,v)
    # u_t = ( (hu)_t - u * h_t ) / h
    h_safe = _np.maximum(h, hmin)
    dudt = (dhu_dt - u * dhdt) / h_safe
    dvdt = (dhv_dt - v * dhdt) / h_safe

    # sanitize NaNs
    dhdt = _np.nan_to_num(dhdt)
    dudt = _np.nan_to_num(dudt)
    dvdt = _np.nan_to_num(dvdt)

    return _np.stack([dhdt, dudt, dvdt], axis=0)

# ----------------------- RHS: Central (fallback) -------------------
def rhs_sw_2d_central(t: float, Y: _np.ndarray, p: _Dict) -> _np.ndarray:
    """
    Older central-difference velocity-form RHS (non-dissipative).
    Kept for reference / A-B comparisons.
    """
    h, u, v = Y[0], Y[1], Y[2]
    g  = float(p.get("g", 9.81))
    f  = float(p.get("f", 0.0))
    nu = float(p.get("nu", 0.0))
    dx = float(p["dx"]); dy = float(p["dy"])
    Du = float(p.get("Du", 0.0)); Dv = float(p.get("Dv", 0.0))
    Fh = p.get("Fh", 0.0)

    hu = h * u
    hv = h * v
    dhdt = -(_ddx(hu, dx) + _ddy(hv, dy))
    if isinstance(Fh, _np.ndarray):
        dhdt = dhdt + Fh
    elif Fh is None:
        pass
    else:
        dhdt = dhdt + float(Fh)

    adv_u = u * _ddx(u, dx) + v * _ddy(u, dy)
    adv_v = u * _ddx(v, dx) + v * _ddy(v, dy)
    pres_x = -g * _ddx(h, dx)
    pres_y = -g * _ddy(h, dy)
    cor_u = -f * v
    cor_v =  f * u
    visc_u = nu * _laplacian(u, dx, dy) if nu else 0.0
    visc_v = nu * _laplacian(v, dx, dy) if nu else 0.0
    damp_u = -Du * u if Du else 0.0
    damp_v = -Dv * v if Dv else 0.0

    dudt = -(adv_u) + cor_u + pres_x + visc_u + damp_u
    dvdt = -(adv_v) + cor_v + pres_y + visc_v + damp_v

    return _np.stack([dhdt, dudt, dvdt], axis=0)

# ----------------------- unified entry point -----------------------
def rhs_sw_2d(t: float, Y: _np.ndarray, p: _Dict) -> _np.ndarray:
    """
    Chooses scheme. Default = 'rusanov'.
    Set p['scheme'] = 'central' to use the old central RHS.
    """
    scheme = str(p.get("scheme", "rusanov")).lower()
    if scheme == "central":
        return rhs_sw_2d_central(t, Y, p)
    return rhs_sw_2d_rusanov(t, Y, p)