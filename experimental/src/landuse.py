import numpy as np

def load_landuse(Ny: int, Nx: int):
    x = np.linspace(0, 1, Nx, endpoint=False)
    band = np.digitize(x, [1/3, 2/3], right=False)  # 0|1|2 left→right
    asphalt = (band == 0); mixed = (band == 1); park = (band == 2)

    albedo = np.where(asphalt, 0.10, np.where(mixed, 0.20, 0.25))
    veg    = np.where(asphalt, 0.00, np.where(mixed, 0.15, 0.60))
    rough  = np.where(asphalt, 0.50, np.where(mixed, 0.35, 0.20))
    return (np.tile(albedo, (Ny, 1)),
            np.tile(veg,    (Ny, 1)),
            np.tile(rough,  (Ny, 1)))