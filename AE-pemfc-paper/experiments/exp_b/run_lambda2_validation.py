"""正式验证式(11)连续化的宽度、求解器和时间容差敏感性。"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from pemfc_coldstart.diagnostics import validate_run
from pemfc_coldstart.io import (create_run_dir, load_configuration,
                                refresh_manifest, save_result)
from pemfc_coldstart.parameters import MODEL_PROTOCOL
from pemfc_coldstart.postprocess import plot_run
from pemfc_coldstart.solver import simulate


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/lambda2_regularization_validation.json"
CASES = {
    "width_0p05_bdf": EXPERIMENT / "config_regularized_005.json",
    "width_0p10_radau": EXPERIMENT / "config_radau.json",
    "width_0p10_tight_bdf": EXPERIMENT / "config_regularized_tight.json",
}
SIGNALS = (
    "voltage_v", "lambda_cathode_mean", "lambda_cathode_max",
    "ice_cathode_mean", "ice_cathode_max", "liquid_cathode_max",
)


def compare(reference_path: Path, candidate_path: Path) -> dict[str, object]:
    reference = np.load(reference_path / "solution.npz")
    candidate = np.load(candidate_path / "solution.npz")
    common_end = min(float(reference["time_s"][-1]), float(candidate["time_s"][-1]))
    mask = reference["time_s"] <= common_end + 1.0e-12
    times = reference["time_s"][mask]
    result: dict[str, object] = {"common_end_s": common_end, "signals": {}}
    for name in SIGNALS:
        baseline = reference[name][mask]
        interpolated = np.interp(times, candidate["time_s"], candidate[name])
        difference = np.abs(interpolated - baseline)
        scale = max(float(np.max(np.abs(baseline))), 1.0e-12)
        result["signals"][name] = {
            "max_abs_difference": float(np.max(difference)),
            "max_relative_to_reference_peak": float(np.max(difference) / scale),
            "final_difference": float(interpolated[-1] - baseline[-1]),
        }
    return result


def main() -> None:
    full_index = json.loads((ROOT / "experiments/full_reproduction_v3_runs.json").read_text())
    reference_path = ROOT / full_index["runs"]["exp_b_isothermal"]["run"]
    records: dict[str, object] = {}
    for key, config in CASES.items():
        print(f"START {key}", flush=True)
        case, params, options, raw = load_configuration(config)
        run_dir = create_run_dir(EXPERIMENT, raw)
        started = time.monotonic()
        result = simulate(case, params, options)
        save_result(run_dir, result, case, params, options, raw, config)
        plot_run(run_dir)
        refresh_manifest(run_dir)
        validation = validate_run(run_dir)
        if not validation["valid"]:
            raise RuntimeError(f"{key}输出完整性检查失败 {validation}")
        records[key] = {
            "run": str(run_dir.relative_to(ROOT)),
            "status": result.status,
            "end_time_s": float(result.time_s[-1]),
            "elapsed_s": time.monotonic() - started,
            "comparison_to_width_0p10_bdf": compare(reference_path, run_dir),
        }
        print(
            f"END {key} status={result.status} t={result.time_s[-1]:.9g}s "
            f"elapsed={records[key]['elapsed_s']:.2f}s",
            flush=True,
        )
    payload = {
        "schema": 1,
        "model_protocol": MODEL_PROTOCOL,
        "reference": str(reference_path.relative_to(ROOT)),
        "runs": records,
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
