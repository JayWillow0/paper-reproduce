"""汇总被文章引用的运行，并计算网格、容差和求解器差异。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from pemfc_coldstart.diagnostics import validate_run
from pemfc_coldstart.io import refresh_manifest


ROOT = Path(__file__).resolve().parents[1]
SELECTED = ROOT / "experiments" / "selected_runs.json"
OUTPUT = ROOT / "experiments" / "summary_metrics.json"


def read_budget(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            value = row["value"]
            try:
                values[row["metric"]] = float(value)
            except ValueError:
                values[row["metric"]] = value
    return values


def run_summary(path: Path) -> dict[str, object]:
    data = np.load(path / "solution.npz")
    metrics = json.loads((path / "metrics.json").read_text())
    return {
        "run_id": path.name,
        "validation": validate_run(path),
        "status": metrics["status"],
        "message": metrics["message"],
        "evaluations": metrics["evaluations"],
        "end_time_s": float(data["time_s"][-1]),
        "voltage_final_v": float(data["voltage_v"][-1]),
        "voltage_min_v": float(np.min(data["voltage_v"])),
        "temperature_max_k": float(np.max(data["temperature_max_k"])),
        "lambda_cathode_mean_final": float(data["lambda_cathode_mean"][-1]),
        "lambda_cathode_max_final": float(data["lambda_cathode_max"][-1]),
        "ice_cathode_mean_final": float(data["ice_cathode_mean"][-1]),
        "ice_cathode_max_final": float(data["ice_cathode_max"][-1]),
        "liquid_cathode_max_final": float(data["liquid_cathode_max"][-1]),
        "charge_predictor_l1_relative_max": float(np.max(data["charge_predictor_l1_relative"])),
        "voltage_rmse_v": metrics.get("voltage_rmse_v"),
        "voltage_mae_v": metrics.get("voltage_mae_v"),
        "budget": read_budget(path / "budgets.csv"),
    }


def compare(candidate: Path, reference: Path) -> dict[str, dict[str, float]]:
    ref = np.load(reference / "solution.npz")
    alt = np.load(candidate / "solution.npz")
    result: dict[str, dict[str, float]] = {}
    for name in ("voltage_v", "lambda_cathode_max", "ice_cathode_max", "liquid_cathode_max"):
        interpolated = np.interp(ref["time_s"], alt["time_s"], alt[name])
        absolute = float(np.max(np.abs(interpolated - ref[name])))
        scale = float(max(np.max(np.abs(ref[name])), 1.0e-12))
        result[name] = {"max_abs": absolute, "max_relative_to_reference_peak": absolute / scale}
    return result


def main() -> None:
    selected_raw = json.loads(SELECTED.read_text())
    selected = {key: ROOT / value for key, value in selected_raw.items()}
    for path in selected.values():
        refresh_manifest(path)
    base = selected["exp_a_isothermal"]
    payload = {
        "schema": 1,
        "selected_runs": {key: run_summary(path) for key, path in selected.items()},
        "numerical_checks_against_exp_a_base": {
            "coarse_grid": compare(selected["exp_a_coarse"], base),
            "fine_grid": compare(selected["exp_a_fine"], base),
            "tight_tolerance": compare(selected["exp_a_tight"], base),
            "radau_solver": compare(selected["exp_a_radau"], base),
        },
        "pore_smoothing_sensitivity": {
            "onset_0.90": {
                "run_id": selected["exp_d_liquid_smooth_090"].name,
                "end_time_s": run_summary(selected["exp_d_liquid_smooth_090"])["end_time_s"],
            },
            "onset_0.95": {
                "run_id": selected["exp_d_liquid"].name,
                "end_time_s": run_summary(selected["exp_d_liquid"])["end_time_s"],
            },
            "onset_0.99": {
                "run_id": selected["exp_d_liquid_smooth_099"].name,
                "end_time_s": run_summary(selected["exp_d_liquid_smooth_099"])["end_time_s"],
            },
        },
        "paper_shutdown_time_digitized_s": {"dissolved": 62.0, "vapor": 48.0, "liquid": 40.0},
        "paper_shutdown_time_uncertainty_s": 0.7,
        "notes": [
            "所有运行均为文献参数与透明补充假设下的直接预测，未对图4拟合。",
            "liquid工况以总液水与冰饱和度达到1的事件报告孔隙堵塞，不等同于电压阈值停机。",
            "动态右端使用恒流反应分布预测器，保存时刻使用完整电势方程校正。",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
