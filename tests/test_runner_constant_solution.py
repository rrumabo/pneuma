import numpy as np
from pathlib import Path
from src.runner import solve_fixed_step

def f_zero(t, u, p):
    return np.zeros_like(u)

def test_runner_keeps_constant_state_and_writes_metrics(tmp_path: Path):
    N = 128
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    u0 = np.sin(2*np.pi*x)  # any field; zero RHS -> stays constant
    dx = 1.0 / N

    sol = solve_fixed_step(
        f_zero,
        t_span=(0.0, 0.01),
        y0=u0,
        dt=1e-3,
        method="rk4",
        save_every=5,
        metrics_out_dir=tmp_path,
        norm_grid=(dx, None),
    )

    assert np.allclose(sol.y[-1], u0, atol=1e-12)
    assert (tmp_path / "metrics.json").exists()
    # meta should expose norms dict (may be empty if norm_grid missing)
    assert "norms" in sol.meta