import numpy as np
from src.core.operators import apply_op

def run_heat_solver_1d(L_op, u0, T, dt, dx=None, step_func=None, rhs_func=None):
    """
    1D heat solver with adapter: works with operator-aware (u,t,dt,L_op,...) or rhs-based (u,rhs,t,dt) steppers.
    """
    if step_func is None:
        raise ValueError("step_func must be provided (e.g. rk4_step_op or rk4_step)")

    u = u0.copy()
    u_history = [u.copy()]
    t = 0.0

    steps = int(np.ceil(T / dt)) if T > 0 else 0

    # Wrap L_op so rhs-based steppers can be used
    def _rhs(u_in, t_in):
        lin = apply_op(L_op, u_in)
        if rhs_func is not None:
            lin = lin + rhs_func(u_in, t_in)
        return lin

    for _ in range(steps):
        try:
            # operator-aware: e.g. rk4_step_op(u, *, t, dt, L_op, rhs_func=None)
            u = step_func(u=u, t=t, dt=dt, L_op=L_op, rhs_func=rhs_func)
        except TypeError:
            # rhs-based: e.g. rk4_step(u, rhs_func, t, dt)
            u = step_func(u, _rhs, t, dt)
        u_history.append(u.copy())
        t += dt

    return u_history