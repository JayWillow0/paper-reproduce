from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .model import ModelParameters, battery_rhs, calculate_reaction_heat_for_node
from .thermal import calculate_core_temperatures, calculate_shell_surface_values


EventFunction = Callable[[float, np.ndarray], float]
SIMULTANEOUS_EVENT_TOLERANCE = 1e-6


def event_values(time_s: float, state: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    n_nodes = parameters.n_nodes
    values = state.reshape(n_nodes, parameters.states_per_node)
    temperatures = values[:, 0]
    energy = values[:, parameters.energy_index]
    core_front, core_back = calculate_core_temperatures(temperatures, parameters)
    output = np.ones(2 * n_nodes + 3, dtype=float)
    trigger = parameters.config.trigger
    if trigger.kind == "needle" and parameters.modes.trigger_short_active:
        output[0] = trigger.release_duration_s - time_s
    for node in range(n_nodes):
        if not parameters.topology.is_battery[node]:
            continue
        cell = int(parameters.topology.cell_index[node])
        core = core_front[cell] if parameters.topology.is_front[node] else core_back[cell]
        if (
            trigger.spontaneous_short_enabled
            and not parameters.modes.spontaneous_short_active[node]
            and not parameters.modes.spontaneous_short_done[node]
        ):
            output[1 + node] = core - trigger.tr_threshold_k
        if parameters.modes.spontaneous_short_active[node]:
            output[1 + n_nodes + node] = trigger.spontaneous_energy_j / 2.0 - energy[node]
    if trigger.kind == "heater" and parameters.modes.heater_active:
        index = 2 * n_nodes + 1
        stop_margins: list[tuple[float, float]] = []
        if trigger.heating_stop_mode == "profile":
            if trigger.heater_profile is None:
                raise RuntimeError("Validated profile heater is missing heater_profile.")
            stop_margins.append(
                (
                    trigger.heater_profile.time_s[-1] - time_s,
                    max(trigger.heater_profile.time_s[-1], 1.0),
                )
            )
        elif trigger.heating_stop_mode == "temperature":
            core_values = []
            for node in parameters.trigger_nodes:
                cell = int(parameters.topology.cell_index[node])
                core_values.append(core_front[cell] if parameters.topology.is_front[node] else core_back[cell])
            stop_margins.append(
                (trigger.tr_threshold_k - max(core_values), trigger.tr_threshold_k)
            )
        else:
            heat = max(calculate_reaction_heat_for_node(state, int(node), parameters) for node in parameters.trigger_nodes)
            cell_mass, cell_cp = parameters.config.cell.effective()
            stop_margins.append(
                (
                    trigger.self_heating_stop_rate_k_per_s - heat / (cell_mass * cell_cp),
                    trigger.self_heating_stop_rate_k_per_s,
                )
            )
        heated_cell = trigger.cell - 1
        if trigger.heater_stop_surface_temperature_k is not None:
            surface_front, surface_back = calculate_shell_surface_values(
                temperatures,
                parameters,
                boundary_value=parameters.config.ambient_temperature_k,
            )
            stop_margins.append(
                (
                    trigger.heater_stop_surface_temperature_k
                    - max(surface_front[heated_cell], surface_back[heated_cell]),
                    trigger.heater_stop_surface_temperature_k,
                )
            )
        if trigger.heater_stop_surface_rate_k_per_s is not None:
            temperature_rate = battery_rhs(time_s, state, parameters).reshape(
                n_nodes, parameters.states_per_node
            )[:, 0]
            rate_front, rate_back = calculate_shell_surface_values(
                temperature_rate,
                parameters,
                boundary_value=0.0,
            )
            stop_margins.append(
                (
                    trigger.heater_stop_surface_rate_k_per_s
                    - max(rate_front[heated_cell], rate_back[heated_cell]),
                    trigger.heater_stop_surface_rate_k_per_s,
                )
            )
        output[index] = min(margin / max(abs(scale), 1e-12) for margin, scale in stop_margins)
    release_temperature = parameters.config.thermal.adiabatic_release_temperature_k
    if parameters.modes.adiabatic_boundary_active and release_temperature is not None:
        cell_centers = []
        for cell in range(parameters.n_batteries):
            front = parameters.topology.node_map[f"bat{cell + 1}_f"]
            back = parameters.topology.node_map[f"bat{cell + 1}_b"]
            cell_centers.append(0.5 * (temperatures[front] + temperatures[back]))
        output[2 * n_nodes + 2] = max(cell_centers) - release_temperature
    return output


def normalized_event_values(
    time_s: float,
    state: np.ndarray,
    parameters: ModelParameters,
) -> np.ndarray:
    """Return dimensionless event surfaces without changing their roots."""
    output = event_values(time_s, state, parameters)
    n_nodes = parameters.n_nodes
    trigger = parameters.config.trigger
    scales = np.ones_like(output)
    scales[0] = max(trigger.release_duration_s, 1.0)
    scales[1 : n_nodes + 1] = max(trigger.tr_threshold_k, 1.0)
    scales[n_nodes + 1 : 2 * n_nodes + 1] = max(
        trigger.spontaneous_energy_j / 2.0, 1.0
    )
    # Heater stop is normalized while its individual margins are assembled.
    release_temperature = parameters.config.thermal.adiabatic_release_temperature_k
    if release_temperature is not None:
        scales[2 * n_nodes + 2] = max(release_temperature, 1.0)
    return output / scales


def active_event_indices(parameters: ModelParameters) -> list[int]:
    """Return zero-based global event indices reachable in the current mode."""
    n_nodes = parameters.n_nodes
    trigger = parameters.config.trigger
    modes = parameters.modes
    indices: list[int] = []
    if trigger.kind == "needle" and modes.trigger_short_active:
        indices.append(0)
    if trigger.spontaneous_short_enabled:
        for node in parameters.battery_nodes:
            node = int(node)
            if not modes.spontaneous_short_active[node] and not modes.spontaneous_short_done[node]:
                indices.append(1 + node)
            if modes.spontaneous_short_active[node]:
                indices.append(1 + n_nodes + node)
    if trigger.kind == "heater" and modes.heater_active:
        indices.append(2 * n_nodes + 1)
    if (
        modes.adiabatic_boundary_active
        and parameters.config.thermal.adiabatic_release_temperature_k is not None
    ):
        indices.append(2 * n_nodes + 2)
    return indices


def build_event_functions(parameters: ModelParameters) -> list[EventFunction]:
    functions: list[EventFunction] = []
    cache_time: float | None = None
    cache_state: np.ndarray | None = None
    cache_state_bytes: bytes | None = None
    cache_values: np.ndarray | None = None

    def cached_values(time_s: float, state: np.ndarray) -> np.ndarray:
        nonlocal cache_time, cache_state, cache_state_bytes, cache_values
        if cache_values is not None and cache_time == time_s and cache_state is state:
            return cache_values
        state_bytes = state.tobytes()
        if cache_values is not None and cache_time == time_s and cache_state_bytes == state_bytes:
            cache_state = state
            return cache_values
        cache_time = time_s
        cache_state = state
        cache_state_bytes = state_bytes
        cache_values = normalized_event_values(time_s, state, parameters)
        return cache_values

    for index in active_event_indices(parameters):
        def event(time_s: float, state: np.ndarray, event_index: int = index) -> float:
            return float(cached_values(time_s, state)[event_index])

        event.terminal = True  # type: ignore[attr-defined]
        event.event_index = index  # type: ignore[attr-defined]
        is_heater_stop = index == 2 * parameters.n_nodes + 1
        if index == 0 or parameters.n_nodes + 1 <= index <= 2 * parameters.n_nodes or is_heater_stop:
            event.direction = -1.0  # type: ignore[attr-defined]
        else:
            event.direction = 1.0  # type: ignore[attr-defined]
        functions.append(event)
    return functions


def collect_event_ids(
    time_s: float,
    state: np.ndarray,
    parameters: ModelParameters,
    reported: Sequence[int],
    *,
    tolerance: float = SIMULTANEOUS_EVENT_TOLERANCE,
) -> list[int]:
    ids = {index + 1 for index in reported}
    values = normalized_event_values(time_s, state, parameters)
    for index in active_event_indices(parameters):
        value = values[index]
        if abs(value) <= tolerance:
            ids.add(index + 1)
    return sorted(ids)


def apply_event_update(parameters: ModelParameters, time_s: float, state: np.ndarray, event_ids: Sequence[int]) -> None:
    n_nodes = parameters.n_nodes
    modes = parameters.modes
    for event_id in event_ids:
        if event_id == 1:
            modes.trigger_short_active = False
        elif 2 <= event_id <= n_nodes + 1:
            node = event_id - 2
            if (
                parameters.config.trigger.spontaneous_short_enabled
                and parameters.topology.is_battery[node]
                and not modes.spontaneous_short_done[node]
            ):
                modes.spontaneous_short_active[node] = True
        elif n_nodes + 2 <= event_id <= 2 * n_nodes + 1:
            node = event_id - n_nodes - 2
            modes.spontaneous_short_active[node] = False
            modes.spontaneous_short_done[node] = True
        elif event_id == 2 * n_nodes + 2:
            modes.heater_active = False
        elif event_id == 2 * n_nodes + 3:
            modes.adiabatic_boundary_active = False
    if parameters.config.trigger.kind == "needle" and time_s >= parameters.config.trigger.release_duration_s:
        modes.trigger_short_active = False
    values = state.reshape(n_nodes, parameters.states_per_node)
    energy = values[:, parameters.energy_index]
    if parameters.config.trigger.spontaneous_short_enabled:
        over_energy = energy >= parameters.config.trigger.spontaneous_energy_j / 2.0
        modes.spontaneous_short_active[over_energy] = False
        modes.spontaneous_short_done[over_energy] = True
    else:
        modes.spontaneous_short_active[:] = False
    temperatures = values[:, 0]
    core_front, core_back = calculate_core_temperatures(temperatures, parameters)
    if not parameters.config.trigger.spontaneous_short_enabled:
        return
    for node in np.flatnonzero(parameters.topology.is_battery):
        if modes.spontaneous_short_active[node] or modes.spontaneous_short_done[node]:
            continue
        cell = int(parameters.topology.cell_index[node])
        core = core_front[cell] if parameters.topology.is_front[node] else core_back[cell]
        if core >= parameters.config.trigger.tr_threshold_k:
            modes.spontaneous_short_active[node] = True


def event_name(event_id: int, n_nodes: int) -> str:
    if event_id == 1:
        return "trigger_short_end"
    if 2 <= event_id <= n_nodes + 1:
        return "spontaneous_short_start"
    if n_nodes + 2 <= event_id <= 2 * n_nodes + 1:
        return "spontaneous_short_end"
    if event_id == 2 * n_nodes + 2:
        return "heater_stop"
    if event_id == 2 * n_nodes + 3:
        return "adiabatic_boundary_release"
    return "unknown"
