from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .config import ModuleConfig, validate_config
from .events import (
    apply_event_update,
    build_event_functions,
    collect_event_ids,
    event_name,
    normalized_event_values,
)
from .model import ModelParameters, battery_rhs, initialize_parameters


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    segment: int
    t_start: float
    t_end: float
    n_points: int
    event_time: float | None
    event_ids: tuple[int, ...]
    solver_method: str
    runtime_warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EventRecord:
    segment: int
    time_s: float
    event_id: int
    name: str
    node_index: int | None
    battery_index: int | None
    node_type: str | None


@dataclass(slots=True)
class SimulationResult:
    t: np.ndarray
    Y: np.ndarray
    params: ModelParameters
    segments: list[SegmentRecord]
    events: list[EventRecord]
    elapsed_s: float
    summary: Any = None

    def to_dict(self, detail: str = "summary") -> dict[str, Any]:
        if detail not in {"summary", "series", "full"}:
            raise ValueError("detail must be 'summary', 'series', or 'full'.")
        payload: dict[str, Any] = {
            "config": self.params.config.to_dict(),
            "summary": self.summary.to_dict(),
            "segments": [asdict(segment) for segment in self.segments],
            "events": [asdict(event) for event in self.events],
        }
        if detail in {"series", "full"}:
            from .postprocess import temperature_series_payload

            payload["temperature_series"] = temperature_series_payload(self)
        if detail == "full":
            payload["time_s"] = self.t.astype(float).tolist()
            payload["state"] = self.Y.astype(float).tolist()
            payload["node_labels"] = list(self.params.topology.labels)
            payload["state_layout"] = ["T", *self.params.concentration_names, "short_energy_released"]
        return payload


def _decode_event(segment: int, time_s: float, event_id: int, parameters: ModelParameters) -> EventRecord:
    n_nodes = parameters.n_nodes
    node: int | None = None
    if 2 <= event_id <= n_nodes + 1:
        node = event_id - 2
    elif n_nodes + 2 <= event_id <= 2 * n_nodes + 1:
        node = event_id - n_nodes - 2
    battery = int(parameters.topology.cell_index[node]) + 1 if node is not None and parameters.topology.is_battery[node] else None
    node_type = parameters.topology.types[node] if node is not None else None
    return EventRecord(segment, time_s, event_id, event_name(event_id, n_nodes), node, battery, node_type)


def _absolute_tolerances(parameters: ModelParameters) -> np.ndarray:
    solver = parameters.config.solver
    template = np.full(parameters.states_per_node, solver.abs_tol_concentration, dtype=float)
    template[0] = solver.abs_tol_temperature
    template[parameters.energy_index] = solver.abs_tol_energy
    return np.tile(template, parameters.n_nodes)


def _jacobian_sparsity(parameters: ModelParameters):
    """Conservative dependency pattern for finite-difference Jacobians."""
    size = parameters.n_nodes * parameters.states_per_node
    pattern = lil_matrix((size, size), dtype=bool)
    for node in range(parameters.n_nodes):
        rows = slice(node * parameters.states_per_node, (node + 1) * parameters.states_per_node)
        columns = slice(node * parameters.states_per_node, (node + 1) * parameters.states_per_node)
        pattern[rows, columns] = True
        # Heat transfer and core-temperature interpolation couple only the
        # immediately adjacent front/back nodes in the one-dimensional stack.
        for neighbor in (parameters.topology.neighbor_front[node], parameters.topology.neighbor_back[node]):
            if neighbor >= 0:
                pattern[rows, neighbor * parameters.states_per_node] = True
    return pattern.tocsr()


def run_simulation(config: ModuleConfig, *, verbose: bool = False) -> SimulationResult:
    config = validate_config(config)
    parameters, state = initialize_parameters(config)
    times: list[np.ndarray] = []
    states: list[np.ndarray] = []
    segments: list[SegmentRecord] = []
    event_records: list[EventRecord] = []
    t0 = 0.0
    start_clock = time.perf_counter()
    solver = config.solver
    atol = _absolute_tolerances(parameters)
    jac_sparsity = _jacobian_sparsity(parameters)
    last_event_signature: tuple[float, tuple[int, ...]] | None = None
    preferred_method = solver.method
    if preferred_method == "auto":
        preferred_method = "LSODA"
    for segment_index in range(1, solver.max_segments + 1):
        heater_event_id = 2 * parameters.n_nodes + 2
        heater_event_index = heater_event_id - 1
        if (
            parameters.modes.heater_active
            and normalized_event_values(t0, state, parameters)[heater_event_index] <= 0.0
        ):
            apply_event_update(parameters, t0, state, [heater_event_id])
            event_records.append(
                _decode_event(segment_index, t0, heater_event_id, parameters)
            )
        event_functions = build_event_functions(parameters)
        attempts = [preferred_method]
        if preferred_method in {"BDF", "Radau"}:
            attempts.append("LSODA")
        solution = None
        actual_method = solver.method
        failures: list[str] = []
        accepted_warnings: tuple[str, ...] = ()
        for method in attempts:
            solve_kwargs: dict[str, Any] = {}
            if method in {"BDF", "Radau"}:
                solve_kwargs["jac_sparsity"] = jac_sparsity
            elif method == "LSODA" and parameters.n_nodes > 2:
                solve_kwargs["lband"] = 2 * parameters.states_per_node
                solve_kwargs["uband"] = 2 * parameters.states_per_node
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("once", RuntimeWarning)
                    candidate = solve_ivp(
                        lambda current_time, current_state: battery_rhs(current_time, current_state, parameters),
                        (t0, config.t_end_s),
                        state,
                        method=method,
                        rtol=solver.rel_tol,
                        atol=atol,
                        max_step=solver.max_step_s,
                        events=event_functions,
                        **solve_kwargs,
                    )
                runtime_warnings = tuple(
                    dict.fromkeys(
                        str(item.message)
                        for item in caught
                        if issubclass(item.category, RuntimeWarning)
                    )
                )
                if not candidate.success:
                    failures.append(f"{method}: {candidate.message}")
                    continue
                if not np.all(np.isfinite(candidate.t)) or not np.all(np.isfinite(candidate.y)):
                    failures.append(f"{method}: solver returned non-finite accepted values")
                    continue
                solution = candidate
                actual_method = method
                accepted_warnings = runtime_warnings
                if method == "LSODA" and preferred_method != "LSODA":
                    preferred_method = "LSODA"
                break
            except (ArithmeticError, RuntimeError, ValueError) as error:
                failures.append(f"{method}: {error}")
        if solution is None:
            raise SimulationError(
                f"All solvers failed in segment {segment_index} from t={t0:.9g} s. " + " | ".join(failures)
            )
        segment_t = solution.t
        segment_y = solution.y.T
        if times:
            segment_t = segment_t[1:]
            segment_y = segment_y[1:, :]
        if segment_t.size:
            times.append(segment_t)
            states.append(segment_y)
        event_positions = [index for index, values in enumerate(solution.t_events) if values.size]
        event_indices = [
            int(event_functions[position].event_index)  # type: ignore[attr-defined]
            for position in event_positions
        ]
        event_time = float(solution.t[-1]) if event_indices else None
        event_ids: list[int] = []
        if event_indices:
            event_ids = collect_event_ids(float(solution.t[-1]), solution.y[:, -1], parameters, event_indices)
        segments.append(
            SegmentRecord(
                segment=segment_index,
                t_start=float(solution.t[0]),
                t_end=float(solution.t[-1]),
                n_points=len(solution.t),
                event_time=event_time,
                event_ids=tuple(event_ids),
                solver_method=actual_method,
                runtime_warnings=accepted_warnings,
            )
        )
        if not event_indices:
            break
        signature = (round(float(solution.t[-1]), 12), tuple(event_ids))
        if signature == last_event_signature:
            raise SimulationError(f"Event loop made no progress at t={solution.t[-1]:.12g} s for events {event_ids}.")
        last_event_signature = signature
        state = solution.y[:, -1].copy()
        energy_positions = np.arange(parameters.energy_index, len(state), parameters.states_per_node)
        energy_tolerance = max(1e-10, float(solver.abs_tol_energy) * 10.0)
        if np.any(state[energy_positions] < -energy_tolerance):
            minimum = float(np.min(state[energy_positions]))
            raise SimulationError(f"Short-energy state violated nonnegative invariant: min={minimum:.9g} J.")
        state[energy_positions] = np.maximum(state[energy_positions], 0.0)
        apply_event_update(parameters, float(solution.t[-1]), state, event_ids)
        for event_id in event_ids:
            event_records.append(_decode_event(segment_index, float(solution.t[-1]), event_id, parameters))
        t0 = float(solution.t[-1])
        if t0 >= config.t_end_s:
            break
    else:
        raise SimulationError(f"Exceeded solver.max_segments={solver.max_segments}; inspect event switching logic.")
    elapsed = time.perf_counter() - start_clock
    result = SimulationResult(
        t=np.concatenate(times) if times else np.array([0.0]),
        Y=np.vstack(states) if states else state.reshape(1, -1),
        params=parameters,
        segments=segments,
        events=event_records,
        elapsed_s=elapsed,
    )
    from .postprocess import summarize_TR_result

    result.summary = summarize_TR_result(result)
    if verbose:
        print(
            f"ModuleTR complete: cells={parameters.n_batteries}, points={len(result.t)}, "
            f"segments={len(segments)}, peak={result.summary.global_peak_temperature_k:.3f} K, elapsed={elapsed:.3f} s"
        )
    return result


def run_TR_model(config: ModuleConfig, run_options: dict[str, Any] | None = None) -> SimulationResult:
    options = run_options or {}
    unknown = set(options) - {"verbose", "max_segments"}
    if unknown:
        raise ValueError(f"Unknown run option(s): {', '.join(sorted(unknown))}.")
    if "max_segments" in options:
        from dataclasses import replace

        config = replace(config, solver=replace(config.solver, max_segments=options["max_segments"]))
    return run_simulation(config, verbose=bool(options.get("verbose", True)))
