from pathlib import Path
from typing import Union
import json
import numpy as np

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).ravel()
    return float(np.sqrt(np.nanmean(d * d)))

def mbe(a: np.ndarray, b: np.ndarray) -> float:
    d = (a - b).ravel()
    return float(np.nanmean(d))

def write_metrics(path: Union[str, Path], data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))