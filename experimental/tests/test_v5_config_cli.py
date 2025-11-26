from pathlib import Path
from src.config import load_config

def test_config_loads_defaults(tmp_path: Path):
    yml = tmp_path/"t.yaml"
    yml.write_text("""model: shallow_water
grid: {Nx: 32, Ny: 24, Lx: 1.0, Ly: 0.75}
time: {dt: 1e-3, T: 0.05}
physics: {g: 9.81, f: 1e-4}
""")
    cfg = load_config(yml)
    assert cfg.grid.Nx==32 and cfg.time.method=="rk4"