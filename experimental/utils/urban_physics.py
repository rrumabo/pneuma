import numpy as np
from numpy.typing import ArrayLike, NDArray

def _as_field(x: ArrayLike, like: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x if x.ndim else np.full_like(like, float(x))

def seb_source(T: np.ndarray, albedo: ArrayLike, veg: ArrayLike, t: float, p: dict) -> NDArray[np.float64]:
    """
    Minimal SEB source:
      net = SW_in*(1 - albedo) - Hc*T - (LEc*veg)*T
      dT/dt = net / C_heat
    SW_in = q0 * max(0, sin(pi * (t mod tau)/tau))
    """
    T = np.asarray(T, dtype=float)
    A = _as_field(albedo, T)
    V = _as_field(veg, T)

    q0  = float(p.get("Q0", 1.0))
    tau = float(p.get("day_len", 1.0))
    C   = float(p.get("C_heat", 1.0))
    Hc  = float(p.get("Hc", 0.5))
    LEc = float(p.get("LEc", 0.4))

    phase = (t % tau) / max(tau, 1e-12)
    sw_in = q0 * max(0.0, np.sin(np.pi * phase))  # scalar
    net   = sw_in * (1.0 - A) - Hc * T - (LEc * V) * T
    dTdt  = net / max(C, 1e-12)
    return np.nan_to_num(dTdt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)

def drag_coeff(roughness: ArrayLike) -> NDArray[np.float64]:
    """
    Linear Rayleigh drag ~ roughness (nondim).
    Always returns an ndarray (0-D if scalar input).
    """
    r = np.asarray(roughness, dtype=float)
    Cd = 1e-3 + 5e-3 * r
    return np.asarray(Cd, dtype=np.float64)