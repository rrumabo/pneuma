import numpy as np
from src.utils.diagnostics import compute_norms

def test_norms_scale_down_monotonic():
    N = 256
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    u = np.sin(2*np.pi*x)
    dx = 1.0 / N

    n1 = compute_norms(u, dx=dx)
    u2 = 0.5 * u
    n2 = compute_norms(u2, dx=dx)

    assert n2["L1"] < n1["L1"]
    assert n2["L2"] < n1["L2"]
    assert n2["Linf"] <= n1["Linf"]