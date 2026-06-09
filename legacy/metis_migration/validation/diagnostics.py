from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np 

def summarize_field(npy_path: str) -> None:
    """Print basic diagnostics for a 2D field stored in a .npy file."""
    path = Path(npy_path).expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    
    field = np.load(path)
    array = np.asarray(field)

    if array.ndim != 2:
        print(f"[Metis] WARNING: expected 2D field, got shape {array.shape}")
    ny, nx = array.shape[-2], array.shape[-1]

    vmin = float(np.min(array))
    vmax = float(np.max(array))
    mean = float(np.mean(array))
    mass = float(np.sum(array)) # "mass" : domain integral (sum) — interpretation depends on variable

    print(f"[Metis] file: {path}")
    print(f"[Metis] shape: {ny} x {nx}")
    print(f"[Metis] min:   {vmin:.4f}")
    print(f"[Metis] max:   {vmax:.4f}")
    print(f"[Metis] mean:  {mean:.4f}")
    print(f"[Metis] mass:  {mass:.4e}")

def main(argv=None) -> None:
    import sys
    args = sys.argv[1:] if argv is None else argv

    if len(args) != 1:
        print("Usage: python -m metis.validation.diagnostics path/to/field.npy")
        raise SystemExit(1)

    summarize_field(args[0])

if __name__ == "__main__":
    main()