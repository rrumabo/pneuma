from src.utils.diagnostics import cfl_number_diffusion

def test_cfl_heat_stable_threshold():
    dx, nu = 1.0/64, 1e-3
    dt = 0.1 * dx*dx / nu   # factor = 0.1  => stable
    factor = cfl_number_diffusion(dt=dt, dx=dx, nu=nu, dim=1)
    assert factor <= 0.5

def test_cfl_heat_violation_detectable():
    dx, nu = 1.0/64, 1e-3
    dt = 0.6 * dx*dx / nu   # factor = 0.6  => unstable
    factor = cfl_number_diffusion(dt=dt, dx=dx, nu=nu, dim=1)
    assert factor > 0.5