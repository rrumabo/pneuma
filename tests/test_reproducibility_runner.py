import numpy as np
from src.runner import solve_fixed_step

def f_linear_decay(t, u, p):
    lam = p.get("lam", 1.0)
    return -lam * u

def test_runner_reproducible_with_same_seed_and_params():
    N = 64
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    u0 = np.cos(2*np.pi*x)

    sol1 = solve_fixed_step(
        f=f_linear_decay,
        t_span=(0.0, 0.05),
        y0=u0,
        dt=1e-3,
        method="rk4",
        params={"lam": 0.25},
        save_every=5,
    )

    sol2 = solve_fixed_step(
        f=f_linear_decay,
        t_span=(0.0, 0.05),
        y0=u0,
        dt=1e-3,
        method="rk4",
        params={"lam": 0.25},
        save_every=5,
    )

    assert sol1.t.shape == sol2.t.shape
    assert np.allclose(sol1.t, sol2.t)
    assert sol1.y.shape == sol2.y.shape
    assert np.allclose(sol1.y, sol2.y, atol=1e-12, rtol=0.0)