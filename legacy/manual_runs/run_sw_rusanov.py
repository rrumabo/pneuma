#!/usr/bin/env python3
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse, json
import numpy as np
from src.models.shallow_water import advance

def main():
    ap = argparse.ArgumentParser("Rusanov SW runner")
    ap.add_argument("--Nx", type=int, default=128)
    ap.add_argument("--Ny", type=int, default=96)
    ap.add_argument("--Lx", type=float, default=1.0)
    ap.add_argument("--Ly", type=float, default=0.75)
    ap.add_argument("--T",  type=float, default=0.2)
    ap.add_argument("--cfl", type=float, default=0.30)
    ap.add_argument("--nu",  type=float, default=1e-4)
    ap.add_argument("--drag",type=float, default=1e-3)
    ap.add_argument("--hmin",type=float, default=1e-6)
    ap.add_argument("--out", type=str, default="outputs/v5_rusanov_run")
    args = ap.parse_args()

    Ny, Nx = args.Ny, args.Nx
    dx, dy  = args.Lx/Nx, args.Ly/Ny
    g = 9.81

    # IC
    x = np.linspace(0,1,Nx,endpoint=False)
    y = np.linspace(0,1,Ny,endpoint=False)
    X, Y = np.meshgrid(x, y)
    h0 = 1.0 + 0.02*np.exp(-((X-0.5)**2+(Y-0.5)**2)/0.02)
    u0 = 0.1*np.sin(2*np.pi*Y)
    v0 = np.zeros_like(u0)

    h,u,v,metrics = advance(h0,u0,v0,dx,dy,
                            t_end=args.T,
                            dt_init=None,
                            g=g, cfl=args.cfl,
                            hmin=args.hmin, nu=args.nu, drag=args.drag)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir/"final_state.npz", h=h, u=u, v=v)
    with open(out_dir/"metrics.json","w") as f: json.dump(metrics, f, indent=2)
    print({"out_dir": str(out_dir), **metrics})

if __name__ == "__main__":
    main()