#!/usr/bin/env python3
"""Validate the pinned upstream checkout and prepare the official demo data link."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "repo" / "TE-method-2024"
EXPECTED_COMMIT = "21b2209508f23d7739e499fd4a90825ce4b3d33f"
DATA_DIR = UPSTREAM / "1-model training" / "Data"
DATA_LINK = ROOT / "experiments" / "data"


def main() -> int:
    if not (UPSTREAM / ".git").is_dir():
        print(f"Upstream Git checkout not found: {UPSTREAM}")
        return 1

    actual = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != EXPECTED_COMMIT:
        print(f"Upstream commit mismatch: expected {EXPECTED_COMMIT}, got {actual}")
        return 1
    if not DATA_DIR.is_dir():
        print(f"Upstream data directory not found: {DATA_DIR}")
        return 1

    relative_target = Path("..") / "repo" / "TE-method-2024" / "1-model training" / "Data"
    if DATA_LINK.is_symlink():
        if os.readlink(DATA_LINK) != str(relative_target):
            print(f"Existing data link points elsewhere: {DATA_LINK}")
            return 1
    elif DATA_LINK.exists():
        print(f"Cannot create data link because the path already exists: {DATA_LINK}")
        return 1
    else:
        DATA_LINK.symlink_to(relative_target)

    print(f"Upstream ready at {EXPECTED_COMMIT}")
    print(f"Demo data link: {DATA_LINK} -> {relative_target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
