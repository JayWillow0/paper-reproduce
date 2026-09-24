from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ModuleConfig
from .parameters import set_config_parameters
from .solver import SimulationResult, run_TR_model, run_simulation


@dataclass(frozen=True, slots=True)
class SensitivityParameter:
    path: str
    values: tuple[float, ...]
    unit: str = ""


@dataclass(frozen=True, slots=True)
class SimulationCase:
    label: str
    config: ModuleConfig
    parameter_name: str = ""
    parameter_value: float | None = None
    unit: str = ""


@dataclass(slots=True)
class SimulationSweepRun:
    case: SimulationCase
    result: SimulationResult
    metrics: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        return {
            "label": self.case.label,
            "parameter_name": self.case.parameter_name,
            "parameter_value": self.case.parameter_value,
            "unit": self.case.unit,
            **self.metrics,
        }


def _positive_workers(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} workers must be a positive integer.")
    return value


def _run_sweep_case(case: SimulationCase) -> SimulationSweepRun:
    try:
        result = run_simulation(case.config, verbose=False)
    except Exception as error:
        raise RuntimeError(f'Simulation sweep case "{case.label}" failed: {error}') from error
    metrics = {
        "global_peak_temperature_k": result.summary.global_peak_temperature_k,
        "cell_tr_time_s": list(result.summary.cell_tr_time_s),
        "propagation_interval_s": list(result.summary.propagation_interval_s),
        "max_propagation_interval_s": result.summary.max_propagation_interval_s,
        "elapsed_s": result.elapsed_s,
        "segment_count": len(result.segments),
    }
    return SimulationSweepRun(case=case, result=result, metrics=metrics)


def run_simulation_sweep(
    cases: Sequence[SimulationCase],
    *,
    verbose: bool = False,
    workers: int = 1,
) -> list[SimulationSweepRun]:
    """Run fully specified cases while retaining each complete simulation result."""
    if not cases:
        raise ValueError("cases cannot be empty.")
    workers = _positive_workers(workers, "simulation sweep")
    for case in cases:
        if not isinstance(case, SimulationCase):
            raise TypeError("cases must contain SimulationCase instances.")
    if workers == 1:
        output = []
        for case in cases:
            result = run_simulation(case.config, verbose=verbose)
            metrics = {
                "global_peak_temperature_k": result.summary.global_peak_temperature_k,
                "cell_tr_time_s": list(result.summary.cell_tr_time_s),
                "propagation_interval_s": list(result.summary.propagation_interval_s),
                "max_propagation_interval_s": result.summary.max_propagation_interval_s,
                "elapsed_s": result.elapsed_s,
                "segment_count": len(result.segments),
            }
            output.append(SimulationSweepRun(case=case, result=result, metrics=metrics))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            output = list(executor.map(_run_sweep_case, cases))
    if verbose:
        for index, case in enumerate(cases, start=1):
            print(f"Sweep {index}/{len(cases)}: {case.label}")
    return output


def _normalize_spec(spec: Sequence[SensitivityParameter | Mapping[str, Any]]) -> list[SensitivityParameter]:
    output: list[SensitivityParameter] = []
    for item in spec:
        if isinstance(item, SensitivityParameter):
            parameter = item
        else:
            raw = dict(item)
            raw["values"] = tuple(raw["values"])
            parameter = SensitivityParameter(**raw)
        if not parameter.path or not parameter.values:
            raise ValueError("Each sensitivity parameter requires a path and at least one value.")
        output.append(parameter)
    if not output:
        raise ValueError("sensitivity_spec cannot be empty.")
    return output


def run_TR_sensitivity(
    base_config: ModuleConfig,
    sensitivity_spec: Sequence[SensitivityParameter | Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    options = dict(options or {})
    allowed = {"mode", "verbose", "run_options", "workers"}
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown sensitivity option(s): {', '.join(sorted(unknown))}.")
    mode = str(options.get("mode", "oat")).lower()
    if mode not in {"oat", "grid"}:
        raise ValueError("sensitivity mode must be 'oat' or 'grid'.")
    spec = _normalize_spec(sensitivity_spec)
    run_options = dict(options.get("run_options", {"verbose": False}))
    workers = _positive_workers(options.get("workers", 1), "sensitivity")
    runs: list[list[tuple[SensitivityParameter, float]]] = []
    if mode == "oat":
        runs = [[(parameter, float(value))] for parameter in spec for value in parameter.values]
    else:
        runs = [list(zip(spec, values, strict=True)) for values in product(*(parameter.values for parameter in spec))]
    payloads = [
        (index, mode, base_config, run, run_options)
        for index, run in enumerate(runs, start=1)
    ]
    if workers == 1:
        output = [_run_sensitivity_payload(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            output = list(executor.map(_run_sensitivity_payload, payloads))
    for index, record in enumerate(output, start=1):
        if options.get("verbose", True):
            print(f"Sensitivity {index}/{len(runs)}: peak={record['global_peak_temperature_k']:.3f} K")
    return output


def _run_sensitivity_payload(
    payload: tuple[
        int,
        str,
        ModuleConfig,
        list[tuple[SensitivityParameter, float]],
        dict[str, Any],
    ],
) -> dict[str, Any]:
    index, mode, base_config, run, run_options = payload
    try:
        config = set_config_parameters(
            base_config,
            [(parameter.path, value) for parameter, value in run],
        )
        result = run_TR_model(config, run_options)
        tr_times = [value for value in result.summary.cell_tr_time_s if value is not None]
        record: dict[str, Any] = {
            "run_index": index,
            "mode": mode,
            "parameter_paths": [parameter.path for parameter, _ in run],
            "parameter_values": [value for _, value in run],
            "global_peak_temperature_k": result.summary.global_peak_temperature_k,
            "first_tr_time_s": min(tr_times) if tr_times else None,
            "max_propagation_interval_s": result.summary.max_propagation_interval_s,
            "max_short_energy_j": result.summary.max_short_energy_j,
            "elapsed_s": result.elapsed_s,
            "segment_count": len(result.segments),
        }
        return record
    except Exception as error:
        paths = ", ".join(parameter.path for parameter, _value in run)
        raise RuntimeError(f"Sensitivity run {index} ({paths}) failed: {error}") from error


def export_sensitivity_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    if not records:
        raise ValueError("records cannot be empty.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: ";".join(map(str, value)) if isinstance(value, list) else value for key, value in record.items()})
