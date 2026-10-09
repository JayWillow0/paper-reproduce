"""核验式(11)C2连续化四图统一重算，并建立当前文章索引。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from pemfc_coldstart.diagnostics import validate_run
from pemfc_coldstart.parameters import MODEL_PROTOCOL


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "experiments" / "full_reproduction_runs.json"
SUMMARY = ROOT / "experiments" / "full_reproduction_summary.json"
V3_INDEX = ROOT / "experiments" / "full_reproduction_v3_runs.json"
REGULARIZATION_VALIDATION = ROOT / "experiments" / "lambda2_regularization_validation.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_budget(path: Path) -> dict[str, object]:
    result: dict[str, object] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                result[row["metric"]] = float(row["value"])
            except ValueError:
                result[row["metric"]] = row["value"]
    return result


def main() -> None:
    v3 = json.loads(V3_INDEX.read_text())
    if v3.get("model_protocol") != MODEL_PROTOCOL:
        raise RuntimeError("统一运行索引的模型协议与当前源码不一致")
    runs = {key: value["run"] for key, value in v3["runs"].items()}
    source_hashes = {
        path.name: sha256(path)
        for path in sorted((ROOT / "src/pemfc_coldstart").glob("*.py"))
    }
    index: dict[str, object] = {}
    summary: dict[str, object] = {
        "schema": 2,
        "protocol": MODEL_PROTOCOL,
        "calibration": "none_direct_prediction",
        "regularization": {
            "mode": "paper_c2_regularized",
            "transition_half_width": 0.10,
            "validation_index": str(REGULARIZATION_VALIDATION.relative_to(ROOT)),
            "historical_literal_index": "experiments/full_reproduction_v2_runs.json",
        },
        "runs": {},
        "diagnostics": {
            "historical_literal_fig5_isothermal_bdf": "failed_at_18.397133413_s_lambda_2_switch",
            "historical_literal_fig5_isothermal_radau": "interrupted_after_54_minutes_at_attracting_lambda_2_switch",
            "radau_incomplete_directory": "experiments/exp_b/runs/20261009T001518Z_7e7f839a",
        },
    }
    for key, relative in runs.items():
        run = ROOT / relative
        validation = validate_run(run)
        manifest = json.loads((run / "manifest.json").read_text())
        metrics = json.loads((run / "metrics.json").read_text())
        resolved = json.loads((run / "params_resolved.json").read_text())
        data = np.load(run / "solution.npz")
        source_matches = manifest["source_sha256"] == source_hashes
        parameters = resolved["parameters"]
        model_matches = (
            manifest.get("model_protocol") == MODEL_PROTOCOL
            and parameters["membrane_diffusivity_mode"] == "paper_c2_regularized"
            and parameters["membrane_diffusivity_transition_half_width"] == 0.10
        )
        if not validation["valid"] or not source_matches or not model_matches:
            raise RuntimeError(f"{key}未通过当前源码与输出完整性核对")
        record = {
            "run": relative,
            "status": metrics["status"],
            "message": metrics["message"],
            "end_time_s": float(data["time_s"][-1]),
            "voltage_final_v": float(data["voltage_v"][-1]),
            "voltage_min_v": float(np.min(data["voltage_v"])),
            "temperature_max_k": float(np.max(data["temperature_max_k"])),
            "lambda_cathode_mean_final": float(data["lambda_cathode_mean"][-1]),
            "ice_cathode_mean_final": float(data["ice_cathode_mean"][-1]),
            "ice_cathode_max_final": float(data["ice_cathode_max"][-1]),
            "liquid_cathode_max_final": float(data["liquid_cathode_max"][-1]),
            "validation": validation,
            "budget": read_budget(run / "budgets.csv"),
            "source_sha256_matches_current": source_matches,
            "model_parameters_match_protocol": model_matches,
            "elapsed_s": v3["runs"][key]["elapsed_s"],
        }
        index[key] = {"run": relative, "status": metrics["status"],
                      "end_time_s": record["end_time_s"]}
        summary["runs"][key] = record
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(INDEX)
    print(SUMMARY)


if __name__ == "__main__":
    main()
