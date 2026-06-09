from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_temperature_field(npy_path: str) -> None:
    """Load a .npy temperature field and show a simple 2D plot."""
    path = Path(npy_path).expanduser().resolve()
    print(f"[Metis] loading {path}")

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    T = np.load(path)
    ny, nx = T.shape
    print(f"[Metis] field shape = {ny}x{nx}, min = {T.min():.2f}, max = {T.max():.2f}")

    plt.figure(figsize=(6, 4))
    im = plt.imshow(T, origin="lower")
    plt.colorbar(im, label="T [°C]")
    plt.title(f"Temperature Field ({ny}x{nx})")
    plt.xlabel("X index")
    plt.ylabel("Y index")
    plt.tight_layout()
    plt.show()


def main(argv=None) -> None:
    import sys
    args = sys.argv[1:] if argv is None else argv

    if len(args) != 1:
        print("Usage: python -m metis.validation.plots path/to/temperature.npy")
        raise SystemExit(1)

    plot_temperature_field(args[0])


if __name__ == "__main__":
    main()
