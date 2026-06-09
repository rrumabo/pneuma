from __future__ import annotations
import sys
from pathlib import Path
from typing import List

from src.config import load_config
from src.runner import run_from_config


def main(argv: List[str] | None = None) -> None:
    """
    Minimal CLI entrypoint.

    Usage (old style, used by tests):
        python -m pneuma.cli --config examples/heat1d.yaml

    Usage (explicit command):
        python -m pneuma.cli --config examples/heat1d.yaml run
        python -m pneuma.cli --config examples/heat1d.yaml validate
    """
    args = list(sys.argv[1:] if argv is None else argv)

    if len(args) < 2 or args[0] != "--config":
        print("Usage: python -m pneuma.cli --config path/to/config.yaml [run|validate]")
        raise SystemExit(1)

    cfg_path = Path(args[1]).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"[pneuma] config not found: {cfg_path}")
        raise SystemExit(1)

    command = "run"
    if len(args) >= 3:
        command = args[2]

    cfg = load_config(cfg_path)

    if command == "validate":
        # if load_config didn't crash, config is OK
        print("Config OK.")
        return

    if command == "run":
        run_from_config(cfg)
        return

    print(f"Unknown command: {command}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()