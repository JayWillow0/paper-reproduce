"""运行预算、范围和保存完整性检查。"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np

from .parameters import MODEL_PROTOCOL, CaseConfig, ModelParameters
from .solver import SimulationResult
from .state import decode_state
from .transport import gas_flux, liquid_flux


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def water_inventory_mol_per_m2(result: SimulationResult, case: CaseConfig,
                                params: ModelParameters) -> np.ndarray:
    values = np.empty(result.time_s.size)
    for k, y in enumerate(result.state.T):
        state = decode_state(y, result.mesh, result.layout, case, params)
        values[k] = np.sum(result.mesh.dx_m * (state.qn + state.qf + state.qv + state.ql + state.qi))
    return values


def budgets(result: SimulationResult, case: CaseConfig, params: ModelParameters) -> dict[str, float | str]:
    inventory = water_inventory_mol_per_m2(result, case, params)
    produced = case.current_a_per_m2 * result.time_s / (2.0 * params.constants.faraday_c_per_mol)
    boundary_rate = np.empty(result.time_s.size)
    for k, vector in enumerate(result.state.T):
        state = decode_state(vector, result.mesh, result.layout, case, params)
        _, anode = gas_flux(state, result.mesh, case, params, "vapor_anode")
        _, cathode = gas_flux(state, result.mesh, case, params, "vapor_cathode")
        _, liquid = liquid_flux(state, result.mesh, params)
        boundary_rate[k] = anode[0] + liquid[0] - cathode[-1] - liquid[-1]
    # 同步累计量与物质状态由同一个积分器推进，避免绘图采样误差污染守恒检验。
    external = float(result.cumulative_budgets[0,-1])
    boundary_integral = external-produced[-1]
    residual = inventory[-1]-inventory[0]-external
    scale = max(abs(inventory[0])+abs(produced[-1])+abs(boundary_integral),1e-12)
    energy_residual = 0.0; energy_scale = 1.0
    if result.layout.thermal:
        energies = result.state[result.layout.slices['energy']]
        delta = float(np.sum((energies[:,-1]-energies[:,0])*result.mesh.dx_m))
        supplied = float(np.sum(result.cumulative_budgets[1:,-1]))
        energy_residual = delta-supplied
        energy_scale = max(abs(delta)+abs(result.cumulative_budgets[1,-1])+abs(result.cumulative_budgets[2,-1]),1.)
    return {
        "energy_budget_residual_j_per_m2": energy_residual if result.layout.thermal else "not_integrated_isothermal",
        "energy_budget_residual_relative": energy_residual/energy_scale if result.layout.thermal else "not_integrated_isothermal",
        "dynamic_charge_residual_max": result.max_dynamic_charge_residual,
        "water_initial_mol_per_m2": float(inventory[0]),
        "water_final_mol_per_m2": float(inventory[-1]),
        "faraday_produced_mol_per_m2": float(produced[-1]),
        "boundary_water_net_mol_per_m2": boundary_integral,
        "water_budget_residual_mol_per_m2": float(residual),
        "water_budget_residual_relative": float(residual / scale),
        "voltage_min_v": float(np.min(result.voltage_v)),
        "temperature_max_k": float(np.max(result.temperature_max_k)),
        "finite_state": str(bool(np.all(np.isfinite(result.state)))),
        "status": result.status,
        "charge_predictor_l1_relative_max": float(np.max(result.charge_predictor_l1_relative)),
    }


def validate_run(run_dir: Path) -> dict[str, object]:
    required = ["manifest.json", "params_resolved.json", "solution.npz", "signals.csv",
                "events.csv", "budgets.csv", "metrics.json", "run.log"]
    missing = [name for name in required if not (run_dir / name).exists()]
    checks: dict[str, object] = {"missing": missing, "valid": not missing}
    if not missing:
        data = np.load(run_dir / "solution.npz")
        time = data["time_s"]
        checks["time_monotone"] = bool(np.all(np.diff(time) > 0) or time.size == 1)
        checks["finite"] = bool(all(np.all(np.isfinite(data[key])) for key in data.files if data[key].dtype.kind in "fc"))
        checks["valid"] = bool(checks["valid"] and checks["time_monotone"] and checks["finite"])
        manifest = json.loads((run_dir / "manifest.json").read_text())
        mismatches = [name for name,value in manifest.get("outputs",{}).items()
                      if not (run_dir/name).is_file() or sha256_file(run_dir/name)!=value]
        checks["hash_mismatches"] = mismatches
        if manifest.get("model_protocol") in {"full_charge_geometric_v2", MODEL_PROTOCOL}:
            checks["physical_domain"] = bool(np.all(data["saved_domain_valid"]))
            checks["dynamic_charge"] = bool(data["max_dynamic_charge_residual"] <=
                json.loads((run_dir/"params_resolved.json").read_text())["solver"]["charge_tolerance"])
            checks["valid"] = bool(checks["valid"] and checks["physical_domain"] and checks["dynamic_charge"])
        checks["valid"] = bool(checks["valid"] and not mismatches)
    return checks


def output_hashes(run_dir: Path) -> dict[str, str]:
    return {str(path.relative_to(run_dir)): sha256_file(path) for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"}
