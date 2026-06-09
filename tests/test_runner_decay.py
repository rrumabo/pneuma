import numpy as np
from src.runner import solve_fixed_step

def f(t, y, p):
    """ODE: y' = -k y"""
    return -p["k"] * y


def test_exponential_decay_rk4_converges():
    """
    y' = -k y with y(0)=1, exact solution y(T)=exp(-kT).
    Check that RK4 global error decreases ~ O(h^4).
    """
    k = 2.0
    T = 2.0
    y0 = np.array([1.0])
    exact = np.exp(-k * T)

    # two different time steps (factor 2 apart)
    dt1 = 1e-2
    dt2 = dt1 / 2

    sol1 = solve_fixed_step(f, t_span=(0.0, T), y0=y0, dt=dt1,
                            method="rk4", params={"k": k})
    sol2 = solve_fixed_step(f, t_span=(0.0, T), y0=y0, dt=dt2,
                            method="rk4", params={"k": k})

    err1 = abs(sol1.y[-1, 0] - exact)
    err2 = abs(sol2.y[-1, 0] - exact)

    # RK4 is 4th order ⇒ halving dt should reduce error by ~16x.
    assert err2 < err1 / 8.0, f"Expected >=8x reduction, got {err1=}, {err2=}"


def test_exponential_decay_euler_basic_correctness():
    """
    Sanity check that Euler produces a decaying solution for y'=-k y.
    """
    k = 1.0
    T = 1.0
    y0 = np.array([1.0])

    sol = solve_fixed_step(f, t_span=(0.0, T), y0=y0, dt=1e-3,
                           method="euler", params={"k": k})

    # Should be positive and less than the initial
    assert sol.y[-1, 0] > 0
    assert sol.y[-1, 0] < y0[0]