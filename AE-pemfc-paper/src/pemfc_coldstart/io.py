"""配置解析、运行目录和结构化输出。"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import csv
import hashlib
import json
import platform
import subprocess
import sys
import numpy as np

from .diagnostics import budgets, output_hashes
from .parameters import (MODEL_PROTOCOL, CaseConfig, ModelParameters, SolverOptions,
                         Material, PhysicalConstants, to_serializable)
from .solver import SimulationResult
from .state import decode_state, validity_margins


def load_configuration(path: Path) -> tuple[CaseConfig, ModelParameters, SolverOptions, dict]:
    raw = json.loads(path.read_text())
    case = CaseConfig(**raw.get("case", {}))
    overrides = dict(raw.get("parameter_overrides", {}))
    if "materials" in overrides:
        overrides["materials"] = tuple(Material(**item) for item in overrides["materials"])
    if "constants" in overrides:
        overrides["constants"] = PhysicalConstants(**overrides["constants"])
    params = replace(ModelParameters(), **overrides)
    options = SolverOptions(**raw.get("solver", {}))
    return case, params, options, raw


def config_hash(raw: dict) -> str:
    payload = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def create_run_dir(experiment: Path, raw: dict) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = experiment / "runs" / f"{stamp}_{config_hash(raw)[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    (path / "figures").mkdir()
    return path


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(header); writer.writerows(rows)


def save_result(run_dir: Path, result: SimulationResult, case: CaseConfig,
                params: ModelParameters, options: SolverOptions, raw: dict,
                config_path: Path) -> None:
    domain_valid = []
    for vector in result.state.T:
        state = decode_state(vector,result.mesh,result.layout,case,params)
        margins = validity_margins(state,result.mesh,case,options.pore_gas_margin,options.pressure_domain_fraction)
        domain_valid.append(all(np.isfinite(value) and value >= -1e-10 for value in margins.values()))
    np.savez_compressed(run_dir / "solution.npz", cumulative_budgets=result.cumulative_budgets,
                        saved_domain_valid=np.asarray(domain_valid),
                        max_dynamic_charge_residual=result.max_dynamic_charge_residual, time_s=result.time_s, state=result.state,
                        x_m=result.mesh.x_m, dx_m=result.mesh.dx_m, layer_name=result.mesh.layer_name,
                        voltage_v=result.voltage_v, temperature_max_k=result.temperature_max_k,
                        lambda_cathode_mean=result.lambda_cathode_mean,
                        lambda_cathode_max=result.lambda_cathode_max,
                        ice_cathode_mean=result.ice_cathode_mean, ice_cathode_max=result.ice_cathode_max,
                        liquid_cathode_mean=result.liquid_cathode_mean,
                        liquid_cathode_max=result.liquid_cathode_max,
                        charge_predictor_l1_relative=result.charge_predictor_l1_relative)
    rows = list(zip(result.time_s, result.voltage_v, result.temperature_max_k,
                    result.lambda_cathode_mean, result.lambda_cathode_max,
                    result.ice_cathode_mean, result.ice_cathode_max,
                    result.liquid_cathode_mean, result.liquid_cathode_max,
                    result.charge_predictor_l1_relative))
    _write_csv(run_dir / "signals.csv", ["time_s", "voltage_v", "temperature_max_k",
               "lambda_cathode_mean", "lambda_cathode_max", "ice_cathode_mean",
               "ice_cathode_max", "liquid_cathode_mean", "liquid_cathode_max",
               "charge_predictor_l1_relative"], rows)
    events = [["integration_end", result.time_s[-1], result.status, result.message]]
    events += [[e["event"],e["time_s"],e["classification"],e["detail"]] for e in result.events]
    for threshold in (0.3, 0.1, 0.05):
        index = np.flatnonzero(result.voltage_v <= threshold)
        if index.size: events.append([f"voltage_below_{threshold:g}_v", result.time_s[index[0]], "down", "sampled"])
    _write_csv(run_dir / "events.csv", ["event", "time_s", "classification", "detail"], events)
    budget = budgets(result, case, params)
    _write_csv(run_dir / "budgets.csv", ["metric", "value"], [[key, value] for key, value in budget.items()])
    metrics = {"model_protocol": MODEL_PROTOCOL,
               "max_dynamic_charge_residual": result.max_dynamic_charge_residual,
               "saved_domain_valid": bool(all(domain_valid)),
               "step_retry_failures": result.failures,
               "trial_domain_extensions": result.trial_domain_extensions,
               "status": result.status, "message": result.message, "evaluations": result.evaluations,
               "calibration_status": "direct_prediction", "reference_comparison": "pending_or_not_available",
               "numerical_convergence": "not_established_by_single_run",
               "charge_predictor_l1_relative_max": float(np.max(result.charge_predictor_l1_relative))}
    reference_files = sorted((config_path.parent / "reference").glob("*experimental_digitized.csv"))
    if reference_files:
        with reference_files[0].open(newline="") as handle:
            reference = list(csv.DictReader(handle))
        ref_t = np.asarray([float(row["time_s"]) for row in reference])
        ref_v = np.asarray([float(row["voltage_v"]) for row in reference])
        mask = ref_t <= result.time_s[-1]
        if np.any(mask):
            predicted = np.interp(ref_t[mask], result.time_s, result.voltage_v)
            metrics["voltage_rmse_v"] = float(np.sqrt(np.mean((predicted - ref_v[mask]) ** 2)))
            metrics["voltage_mae_v"] = float(np.mean(np.abs(predicted - ref_v[mask])))
            metrics["reference_points_covered"] = int(np.sum(mask))
            metrics["reference_points_total"] = int(ref_t.size)
            metrics["reference_comparison"] = "digitized_experiment"
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    resolved = {"case": to_serializable(case), "parameters": to_serializable(params),
                "solver": to_serializable(options), "source_config": str(config_path)}
    (run_dir / "params_resolved.json").write_text(json.dumps(resolved, ensure_ascii=False, indent=2) + "\n")
    (run_dir / "run.log").write_text(f"status={result.status}\nmessage={result.message}\nevaluations={result.evaluations}\n")
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    except Exception:
        freeze = []
    source_root = Path(__file__).resolve().parent
    source_hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in sorted(source_root.glob("*.py"))}
    manifest = {"model_protocol": MODEL_PROTOCOL, "source_sha256": source_hashes,
                "run_id": run_dir.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "config_sha256": config_hash(raw), "python": sys.version, "platform": platform.platform(),
                "dependencies": freeze, "outputs": output_hashes(run_dir)}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def refresh_manifest(run_dir: Path) -> None:
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["outputs"] = output_hashes(run_dir)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
