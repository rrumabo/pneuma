import numpy as np

def _ddx_central(f: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2.0 * dx)

def _ddy_central(f: np.ndarray, dy: float) -> np.ndarray:
    return (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2.0 * dy)

def geostrophic_uv_from_h(h: np.ndarray, g: float, f: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Geostrophic balance on an f-plane:
        u_g = -(g/f) * ∂h/∂y
        v_g =  +(g/f) * ∂h/∂x
    Assumes periodic BCs and co-located variables.
    """
    if abs(f) < 1e-8:
        raise ValueError("Coriolis parameter f is too small for geostrophic balance.")
    dhdx = _ddx_central(h, dx)
    dhdy = _ddy_central(h, dy)
    u_g = -(g / f) * dhdy
    v_g = +(g / f) * dhdx
    return u_g, v_g

def box_smooth(a: np.ndarray, iters: int = 0) -> np.ndarray:
    """Optional tiny smoothing (3x3 box) to tame noise without new deps."""
    out = a.copy()
    k = 1
    for _ in range(max(0, iters)):
        # 3x3 periodic average
        out = (
            out
            + np.roll(out,  1, 0) + np.roll(out, -1, 0)
            + np.roll(out,  1, 1) + np.roll(out, -1, 1)
            + np.roll(np.roll(out,  1, 0),  1, 1)
            + np.roll(np.roll(out,  1, 0), -1, 1)
            + np.roll(np.roll(out, -1, 0),  1, 1)
            + np.roll(np.roll(out, -1, 0), -1, 1)
        ) / 9.0
    return out