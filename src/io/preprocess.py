from pathlib import Path
from typing import Any

def preproc_era5_to_npz(config_path: str | Path, *args: Any, **kwargs: Any) -> None:
    """
    Minimal stub for ERA5 preprocessing.

    The real implementation has been moved to experimental/ and is NOT part of
    the core v1 PDE sandbox. If you need ERA5 preproc, wire it explicitly from
    experimental/ and add tests for it.
    """
    raise SystemExit(
        "[pneuma] ERA5 preprocessing is not part of the core v1. "
        "Use experimental/scripts/get_era5_cyprus.py or implement a proper "
        "preproc_era5_to_npz with tests before exposing it in the CLI."
    )