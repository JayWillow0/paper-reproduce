"""命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

from .diagnostics import validate_run
from .io import create_run_dir, load_configuration, refresh_manifest, save_result
from .postprocess import plot_run
from .solver import simulate


def main() -> int:
    parser = argparse.ArgumentParser(prog="pemfc-coldstart")
    sub = parser.add_subparsers(dest="command", required=True)
    simulate_parser = sub.add_parser("simulate")
    simulate_parser.add_argument("--config", type=Path, required=True)
    simulate_parser.add_argument("--output", type=Path)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--run", type=Path, required=True)
    plot_parser = sub.add_parser("plot")
    plot_parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "simulate":
        case, params, options, raw = load_configuration(args.config)
        run_dir = args.output or create_run_dir(args.config.parent, raw)
        run_dir.mkdir(parents=True, exist_ok=True); (run_dir / "figures").mkdir(exist_ok=True)
        result = simulate(case, params, options)
        save_result(run_dir, result, case, params, options, raw, args.config)
        plot_run(run_dir)
        refresh_manifest(run_dir)
        print(run_dir)
        return 0 if result.status != "failed" else 2
    if args.command == "validate":
        report = validate_run(args.run); print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 3
    plot_run(args.run); print(args.run / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
