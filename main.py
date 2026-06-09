import argparse, os, numpy as np
from src.runner import solve_fixed_step
from src.utils.config_loader import load_config
from src.pdes.heat_solver_1d import run_heat_solver_1d 
from src.visualization.plotting_1_2d import plot_field_1d

def rhs_heat_1d(t, u, p):
    """
    RHS for the 1D heat equation: u_t = alpha * Laplacian(u).
    p must contain:
      - "L_op": discrete Laplacian operator (matrix or callable)
      - "alpha": diffusion coefficient
    """
    L_op = p["L_op"]
    alpha = p.get("alpha", 1.0)
    # Support either a callable operator or a matrix
    if callable(L_op):
        Lu = L_op(u)
    else:
        Lu = L_op @ u
    return alpha * Lu

def build_rhs(cfg):
    """
    Ensure physics params contain an L_op for the 1D heat equation.
    If missing, build a simple periodic Laplacian as a callable using the grid.
    Returns an RHS function f(t,u,p).
    """
    p = cfg["physics"]["params"]

    # If no operator is provided in the config, create a periodic 1D Laplacian
    if "L_op" not in p:
        N_raw, L_raw = cfg["grid"]["N"], cfg["grid"]["L"]
        N = int(N_raw if isinstance(N_raw, (int, float)) else N_raw[0])
        L = float(L_raw if isinstance(L_raw, (int, float)) else L_raw[0])
        dx = L / N

        def L_op(u):
            # periodic second difference
            return (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)

        p["L_op"] = L_op

    # Return the RHS closure expected by the runner
    return lambda t, u, params: rhs_heat_1d(t, u, params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rhs = build_rhs(cfg)
    dt, T = cfg["time"]["dt"], cfg["time"]["T"]
    N, L = cfg["grid"]["N"], cfg["grid"]["L"]
    outdir = cfg["io"]["outdir"]

    # grid + IC (1D)
    N = int(N if isinstance(N, (int, float)) else N[0])
    L = float(L if isinstance(L, (int, float)) else L[0])
    x = np.linspace(0, L, N, endpoint=False)

    # simple Gaussian IC (extend later to presets)
    ic = cfg["initial_condition"]
    c = ic["center"]; c = c[0] if isinstance(c, list) else c
    s = ic["sigma"];  s = s[0] if isinstance(s, list) else s
    amp = ic["amp"]
    u0 = amp * np.exp(-((x - c) % L)**2 / (2*s*s))

    sol = solve_fixed_step(rhs, t_span=(0.0, T), y0=u0, dt=dt,
                           method=cfg["integrator"]["method"],
                           params=cfg["physics"]["params"], save_every=10)

    os.makedirs(outdir, exist_ok=True)
    np.savetxt(os.path.join(outdir, "x.csv"), x, delimiter=",")
    np.savetxt(os.path.join(outdir, "u_final.csv"), sol.y[-1], delimiter=",")
    plot_field_1d(x, sol.y[-1], outdir=outdir, filename="u_final.png", title="Final field")
    print("Done:", sol.meta)

if __name__ == "__main__":
    main()