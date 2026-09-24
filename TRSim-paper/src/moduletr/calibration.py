from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import re
import time
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .config import ModuleConfig
from .parameters import set_config_parameters
from .postprocess import core_temperature_series, extract_shell_surface_temperatures
from .solver import SimulationError, SimulationResult, run_TR_model


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class FitParameter:
    path: str
    initial: float
    lower_bound: float
    upper_bound: float
    scale: float = 1.0
    unit: str = ""
    enabled: bool = True

    def validate(self) -> None:
        values = np.asarray([self.initial, self.lower_bound, self.upper_bound, self.scale], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Fit parameter {self.path} contains non-finite values.")
        if self.lower_bound >= self.upper_bound:
            raise ValueError(f"Fit parameter {self.path} requires lower_bound < upper_bound.")
        if not self.lower_bound <= self.initial <= self.upper_bound:
            raise ValueError(f"Fit parameter {self.path} initial value must lie inside its bounds.")
        if self.scale <= 0:
            raise ValueError(f"Fit parameter {self.path} scale must be > 0.")


@dataclass(slots=True)
class CalibrationResult:
    best_config: ModuleConfig
    best_params: dict[str, float]
    best_score: float
    algorithm_used: str
    history: list[dict[str, Any]]
    sim_result: SimulationResult

    def to_dict(self, include_simulation: bool = False) -> dict[str, Any]:
        payload = {
            "best_config": self.best_config.to_dict(),
            "best_params": self.best_params,
            "best_score": self.best_score,
            "algorithm_used": self.algorithm_used,
            "history": self.history,
        }
        if include_simulation:
            payload["simulation"] = self.sim_result.to_dict("series")
        return payload


@dataclass(frozen=True, slots=True)
class _CalibrationProblem:
    base_config: ModuleConfig
    target_time: np.ndarray
    target_signals: dict[str, np.ndarray]
    parameters: tuple[FitParameter, ...]
    scale: np.ndarray
    weights: dict[str, float]
    run_options: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _EvaluationResult:
    values: np.ndarray
    score: float
    status: str
    error_type: str | None = None
    error_message: str | None = None


def _apply_candidate(problem: _CalibrationProblem, values: np.ndarray) -> ModuleConfig:
    physical_values = np.asarray(values, dtype=float) * problem.scale
    return set_config_parameters(
        problem.base_config,
        [
            (item.path, float(value))
            for item, value in zip(problem.parameters, physical_values, strict=True)
        ],
    )


def _evaluate_candidate(problem: _CalibrationProblem, values: np.ndarray) -> _EvaluationResult:
    values = np.asarray(values, dtype=float)
    try:
        simulation = run_TR_model(_apply_candidate(problem, values), problem.run_options)
        total = 0.0
        total_weight = 0.0
        for name, target in problem.target_signals.items():
            prediction = _linear_interp_extrapolate(
                simulation.t,
                _extract_signal(simulation, name),
                problem.target_time,
            )
            weight = float(problem.weights.get(name, 1.0))
            valid = np.isfinite(target)
            total += weight * float(np.mean((prediction[valid] - target[valid]) ** 2))
            total_weight += weight
        score = float(np.sqrt(total / total_weight))
        if not np.isfinite(score):
            raise FloatingPointError("Calibration objective produced a non-finite score.")
        return _EvaluationResult(values.copy(), score, "ok")
    except (ValueError, SimulationError, ArithmeticError) as error:
        return _EvaluationResult(
            values.copy(),
            1e12,
            "failed",
            type(error).__name__,
            str(error),
        )


def _evaluate_candidate_payload(
    payload: tuple[_CalibrationProblem, np.ndarray],
) -> _EvaluationResult:
    return _evaluate_candidate(*payload)


class _CalibrationProcessMap:
    def __init__(
        self,
        executor: ProcessPoolExecutor,
        problem: _CalibrationProblem,
        record: Callable[[_EvaluationResult], None],
    ) -> None:
        self.executor = executor
        self.problem = problem
        self.record = record

    def __call__(self, _function: Callable[..., float], iterable: Any) -> list[float]:
        candidates = [np.asarray(values, dtype=float) for values in iterable]
        results = list(
            self.executor.map(
                _evaluate_candidate_payload,
                [(self.problem, values) for values in candidates],
            )
        )
        for result in results:
            self.record(result)
        return [result.score for result in results]


def _normalize_fit_parameters(spec: Sequence[FitParameter | Mapping[str, Any]]) -> list[FitParameter]:
    output: list[FitParameter] = []
    for item in spec:
        if isinstance(item, FitParameter):
            parameter = item
        else:
            raw = dict(item)
            aliases = {"lb": "lower_bound", "ub": "upper_bound"}
            for source, target in aliases.items():
                if source in raw:
                    raw[target] = raw.pop(source)
            parameter = FitParameter(**raw)
        parameter.validate()
        if parameter.enabled:
            output.append(parameter)
    if not output:
        raise ValueError("fit_spec must contain at least one enabled parameter.")
    return output


def _normalize_target(target_data: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not isinstance(target_data, Mapping) or "time_s" not in target_data:
        raise ValueError("target_data must be a mapping containing time_s.")
    time_s = np.asarray(target_data["time_s"], dtype=float).reshape(-1)
    if time_s.size < 2 or not np.all(np.isfinite(time_s)) or np.any(np.diff(time_s) <= 0):
        raise ValueError("target_data.time_s must be finite and strictly increasing with at least two points.")
    signals: dict[str, np.ndarray] = {}
    for name, values in target_data.items():
        if name == "time_s":
            continue
        array = np.asarray(values, dtype=float).reshape(-1)
        if len(array) != len(time_s) or np.any(np.isinf(array)):
            raise ValueError(f"Target signal {name} must match time_s and cannot contain infinity.")
        if np.count_nonzero(np.isfinite(array)) < 2:
            raise ValueError(f"Target signal {name} must contain at least two finite values.")
        signals[name] = array
    if not signals:
        raise ValueError("target_data must contain at least one temperature signal.")
    return time_s, signals


_TARGET_SIGNAL_PATTERN = re.compile(
    r"T_cell(\d+)_(center|front|back|surface_front|surface_back)_K"
)


def _validate_target_signal_names(config: ModuleConfig, names: Sequence[str]) -> None:
    for name in names:
        match = _TARGET_SIGNAL_PATTERN.fullmatch(name)
        if not match:
            raise ValueError(
                f'Unsupported target signal "{name}"; use T_cellN_center_K/front_K/back_K '
                "or T_cellN_surface_front_K/surface_back_K."
            )
        cell = int(match.group(1))
        if not 1 <= cell <= config.topology.n_cells:
            raise ValueError(f"Signal {name} refers to a cell outside the configured module.")


def _extract_signal(result: SimulationResult, name: str) -> np.ndarray:
    match = _TARGET_SIGNAL_PATTERN.fullmatch(name)
    if not match:
        raise ValueError(
            f'Unsupported target signal "{name}"; use T_cellN_center_K/front_K/back_K '
            "or T_cellN_surface_front_K/surface_back_K."
        )
    cell = int(match.group(1)) - 1
    if not 0 <= cell < result.params.n_batteries:
        raise ValueError(f"Signal {name} refers to a cell outside the configured module.")
    location = match.group(2)
    if location.startswith("surface_"):
        front, back = extract_shell_surface_temperatures(result.Y, result.params)
        return {"surface_front": front, "surface_back": back}[location][:, cell]
    front, back, center = core_temperature_series(result.Y, result.params)
    return {"front": front, "back": back, "center": center}[location][:, cell]


def _linear_interp_extrapolate(x: np.ndarray, y: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = np.interp(target, x, y)
    left = target < x[0]
    right = target > x[-1]
    if np.any(left):
        output[left] = y[0] + (target[left] - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    if np.any(right):
        output[right] = y[-1] + (target[right] - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2])
    return output


def run_TR_calibration(
    base_config: ModuleConfig,
    target_data: Mapping[str, Sequence[float]],
    fit_spec: Sequence[FitParameter | Mapping[str, Any]],
    options: Mapping[str, Any] | None = None,
) -> CalibrationResult:
    options = dict(options or {})
    allowed = {
        "algorithm",
        "max_iter",
        "population_size",
        "random_starts",
        "verbose",
        "progress",
        "progress_every",
        "progress_heartbeat_s",
        "progress_label",
        "rng_seed",
        "weights",
        "run_options",
        "workers",
    }
    unknown = set(options) - allowed
    if unknown:
        raise ValueError(f"Unknown calibration option(s): {', '.join(sorted(unknown))}.")
    algorithm = str(options.get("algorithm", "auto")).lower()
    if algorithm == "particleswarm":
        raise ValueError("'particleswarm' is not emulated; use algorithm='auto' for differential evolution plus Nelder-Mead.")
    if algorithm not in {"auto", "ga", "fminsearch", "random", "differential_evolution", "nelder_mead"}:
        raise ValueError(f'Unknown calibration algorithm "{algorithm}".')
    parameters = _normalize_fit_parameters(fit_spec)
    target_time, target_signals = _normalize_target(target_data)
    _validate_target_signal_names(base_config, tuple(target_signals))
    weights = dict(options.get("weights", {}))
    unknown_weights = set(weights) - set(target_signals)
    if unknown_weights:
        raise ValueError(f"Weights contain unknown signal(s): {', '.join(sorted(unknown_weights))}.")
    scale = np.asarray([item.scale for item in parameters], dtype=float)
    lower = np.asarray([item.lower_bound for item in parameters], dtype=float) / scale
    upper = np.asarray([item.upper_bound for item in parameters], dtype=float) / scale
    initial = np.asarray([item.initial for item in parameters], dtype=float) / scale
    history: list[dict[str, Any]] = []
    run_options = dict(options.get("run_options", {"verbose": False}))
    workers = options.get("workers", 1)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError("calibration workers must be a positive integer.")
    show_progress = bool(options.get("progress", False))
    progress_every = int(options.get("progress_every", 1))
    if progress_every <= 0:
        raise ValueError("calibration progress_every must be a positive integer.")
    progress_label = str(options.get("progress_label", "calibration"))
    progress_heartbeat_s = float(options.get("progress_heartbeat_s", 0.0))
    if not np.isfinite(progress_heartbeat_s) or progress_heartbeat_s < 0.0:
        raise ValueError("calibration progress_heartbeat_s must be finite and >= 0.")
    progress_start = time.perf_counter()
    best_seen = float("inf")
    successful_evaluations = 0

    problem = _CalibrationProblem(
        base_config=base_config,
        target_time=target_time,
        target_signals=target_signals,
        parameters=tuple(parameters),
        scale=scale,
        weights=weights,
        run_options=run_options,
    )

    def apply(values: np.ndarray) -> ModuleConfig:
        return _apply_candidate(problem, values)

    def record_evaluation(evaluation: _EvaluationResult) -> None:
        nonlocal best_seen, successful_evaluations
        record: dict[str, Any] = {
            "iteration": len(history) + 1,
            "score": evaluation.score,
            "status": evaluation.status,
        }
        physical_values = evaluation.values * scale
        record.update(
            {
                item.path: float(value)
                for item, value in zip(parameters, physical_values, strict=True)
            }
        )
        if evaluation.status == "failed":
            record["error_type"] = evaluation.error_type
            record["error_message"] = evaluation.error_message
        else:
            successful_evaluations += 1
        history.append(record)
        best_seen = min(best_seen, evaluation.score)
        if show_progress and (len(history) == 1 or len(history) % progress_every == 0):
            elapsed = time.perf_counter() - progress_start
            print(
                f"[{progress_label}] evaluation {len(history)}: "
                f"RMSE={evaluation.score:.4g} K, best={best_seen:.4g} K, elapsed={elapsed:.1f} s",
                flush=True,
            )

    def objective(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if np.any(values < lower) or np.any(values > upper):
            distance = np.maximum(lower - values, 0) + np.maximum(values - upper, 0)
            return float(1e6 + np.dot(distance, distance))
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        try:
            if show_progress and progress_heartbeat_s > 0.0:
                evaluation_number = len(history) + 1

                def heartbeat() -> None:
                    while not heartbeat_stop.wait(progress_heartbeat_s):
                        elapsed = time.perf_counter() - progress_start
                        print(
                            f"[{progress_label}] evaluation {evaluation_number} is still running; "
                            f"total elapsed={elapsed:.1f} s",
                            flush=True,
                        )

                heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
                heartbeat_thread.start()
            evaluation = _evaluate_candidate(problem, values)
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=0.1)
        record_evaluation(evaluation)
        return evaluation.score

    max_iter = int(options.get("max_iter", 20))
    seed = int(options.get("rng_seed", 1))
    population_size = int(options.get("population_size", max(10, 4 * len(parameters))))
    if algorithm in {"auto", "ga", "differential_evolution"}:
        popsize = max(1, int(np.ceil(population_size / len(parameters))))
        if workers == 1:
            global_result = differential_evolution(
                objective,
                list(zip(lower, upper, strict=True)),
                maxiter=max_iter,
                popsize=popsize,
                seed=seed,
                polish=False,
                updating="immediate",
            )
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                process_map = _CalibrationProcessMap(executor, problem, record_evaluation)
                global_result = differential_evolution(
                    objective,
                    list(zip(lower, upper, strict=True)),
                    maxiter=max_iter,
                    popsize=popsize,
                    seed=seed,
                    polish=False,
                    updating="deferred",
                    workers=process_map,
                )
        if algorithm == "auto":
            local = minimize(objective, global_result.x, method="Nelder-Mead", options={"maxiter": max_iter})
            best_values = local.x if local.fun <= global_result.fun else global_result.x
            best_score = min(float(local.fun), float(global_result.fun))
            algorithm_used = "differential_evolution+Nelder-Mead"
        else:
            best_values, best_score = global_result.x, float(global_result.fun)
            algorithm_used = "scipy.differential_evolution"
    else:
        rng = np.random.default_rng(seed)
        starts = [initial]
        for _ in range(max(0, int(options.get("random_starts", max(4, 2 * len(parameters)))) - 1)):
            starts.append(rng.uniform(lower, upper))
        results = [minimize(objective, start, method="Nelder-Mead", options={"maxiter": max_iter}) for start in starts]
        best = min(results, key=lambda item: item.fun)
        best_values, best_score = best.x, float(best.fun)
        algorithm_used = "scipy.Nelder-Mead_multistart"
    if successful_evaluations == 0:
        reasons = sorted(
            {
                f"{record.get('error_type')}: {record.get('error_message')}"
                for record in history
                if record.get("status") == "failed"
            }
        )
        detail = " | ".join(reasons[:3]) if reasons else "no candidate completed successfully"
        raise CalibrationError(f"All calibration evaluations failed: {detail}")
    best_config = apply(np.clip(best_values, lower, upper))
    final_run_options = dict(run_options)
    final_run_options["verbose"] = bool(options.get("verbose", False))
    try:
        simulation = run_TR_model(best_config, final_run_options)
    except (ValueError, SimulationError, ArithmeticError) as error:
        raise CalibrationError(f"Final calibrated simulation failed: {error}") from error
    return CalibrationResult(
        best_config=best_config,
        best_params={
            item.path: float(value)
            for item, value in zip(
                parameters,
                np.clip(best_values, lower, upper) * scale,
                strict=True,
            )
        },
        best_score=best_score,
        algorithm_used=algorithm_used,
        history=history,
        sim_result=simulation,
    )
