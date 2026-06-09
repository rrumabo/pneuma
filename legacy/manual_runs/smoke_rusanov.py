#!/usr/bin/env python3
import sys
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, json
from src.models.shallow_water import advance

# grid & params
Ny, Nx = 96, 128
Lx, Ly = 1.0, 0.75
dx, dy = Lx/Nx, Ly/Ny
g = 9.81

# simple IC
x = np.linspace(0,1,Nx,endpoint=False)
y = np.linspace(0,1,Ny,endpoint=False)
X, Y = np.meshgrid(x, y)
h0 = 1.0 + 0.02*np.exp(-((X-0.5)**2+(Y-0.5)**2)/0.02)
u0 = 0.1*np.sin(2*np.pi*Y)
v0 = np.zeros_like(u0)

# run with adaptive CFL
h,u,v,metrics = advance(h0,u0,v0,dx,dy,
                        t_end=0.2,
                        dt_init=None,
                        g=g, cfl=0.3,
                        hmin=1e-6, nu=1e-4, drag=1e-3)

out_dir = Path("outputs/v5_rusanov_smoke"); out_dir.mkdir(parents=True, exist_ok=True)
np.savez(out_dir/"final_state.npz", h=h, u=u, v=v)
with open(out_dir/"metrics.json","w") as f: json.dump(metrics, f, indent=2)
print("OK ->", out_dir)
print(metrics)


def total_mass(h, dx, dy): 
    return float(h.sum()) * dx * dy

M0 = total_mass(h0, dx, dy)
M1 = total_mass(h,  dx, dy)
rel_drift = abs(M1 - M0) / max(M0, 1e-12)
print({"mass_rel_drift": rel_drift})

plt.figure(); plt.imshow(h, origin="lower"); plt.colorbar(label="h")
plt.title("Final h (Rusanov)"); plt.tight_layout()
Path(out_dir).mkdir(parents=True, exist_ok=True)
plt.savefig(Path(out_dir)/"h_final_rusanov.png", dpi=140); plt.show()