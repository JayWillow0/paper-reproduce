from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, TYPE_CHECKING

import numpy as np

from .reactions import concentration_rates_batch, reaction_heat_components_batch

if TYPE_CHECKING:
    from .model import ModelParameters
    from .solver import SimulationResult


@dataclass(frozen=True, slots=True)
class HeatPowerSeries:
    time_s: np.ndarray
    reaction_components_w: dict[str, np.ndarray]
    reaction_total_w: np.ndarray
    short_circuit_w: np.ndarray
    external_heater_w: np.ndarray
    generated_w: np.ndarray
    transfer_w: np.ndarray
    net_w: np.ndarray
    temperature_rate_k_per_s: np.ndarray
    parameters: ModelParameters

    def cell_sum(self, values: np.ndarray, cell_number: int) -> np.ndarray:
        if not 1 <= cell_number <= self.parameters.n_batteries:
            raise ValueError(f"cell_number must be in [1, {self.parameters.n_batteries}].")
        front = self.parameters.node_map[f"bat{cell_number}_f"]
        back = self.parameters.node_map[f"bat{cell_number}_b"]
        return np.asarray(values)[:, front] + np.asarray(values)[:, back]

    def module_sum(self, values: np.ndarray, *, include_holders: bool = False) -> np.ndarray:
        array = np.asarray(values)
        mask = np.ones(self.parameters.n_nodes, dtype=bool) if include_holders else self.parameters.topology.is_battery
        return np.sum(array[:, mask], axis=1)


def reconstruct_heat_power(result: SimulationResult) -> HeatPowerSeries:
    """Recompute accepted-state heat terms; no RHS logging or runtime mutation."""
    parameters = result.params
    n_time = len(result.t)
    n_nodes = parameters.n_nodes
    values = result.Y.reshape(n_time, n_nodes, parameters.states_per_node)
    names = parameters.reaction_names
    components = {name: np.zeros((n_time, n_nodes), dtype=float) for name in names}
    transfer = np.zeros((n_time, n_nodes), dtype=float)
    release_times = [event.time_s for event in result.events if event.name == "adiabatic_boundary_release"]
    release_time = min(release_times) if release_times else None
    temperatures = values[:, :, 0]
    core_front, core_back, _center = core_temperature_series(result.Y, parameters)
    ambient_coeff = np.broadcast_to(
        parameters.ambient_coeff_initial,
        (n_time, n_nodes),
    ).copy()
    if release_time is not None:
        ambient_coeff[result.t >= release_time] = parameters.ambient_coeff_after_release
    transfer[:] = (parameters.conductance @ temperatures.T).T + ambient_coeff * (
        parameters.config.ambient_temperature_k - temperatures
    )
    for battery_offset, node in enumerate(parameters.battery_nodes):
        node = int(node)
        cell = int(parameters.battery_cells[battery_offset])
        core = core_front[:, cell] if parameters.battery_is_front[battery_offset] else core_back[:, cell]
        node_components = reaction_heat_components_batch(
            temperatures[:, node],
            core,
            values[:, node, parameters.concentration_slice],
            parameters.prepared_reactions,
        )
        for name, heat in node_components.items():
            components[name][:, node] = heat
    reaction_total = np.sum(np.stack(tuple(components.values()), axis=0), axis=0)
    short = np.zeros((n_time, n_nodes), dtype=float)
    trigger = parameters.config.trigger
    if trigger.kind == "needle":
        mask = result.t < min(trigger.release_duration_s, result.t[-1] + 1.0)
        short[np.ix_(mask, parameters.trigger_nodes)] = parameters.trigger_power_w
    starts: dict[int, float] = {}
    intervals: dict[int, list[tuple[float, float]]] = {}
    for event in result.events:
        if event.node_index is None:
            continue
        if event.name == "spontaneous_short_start":
            starts[event.node_index] = event.time_s
        elif event.name == "spontaneous_short_end" and event.node_index in starts:
            intervals.setdefault(event.node_index, []).append((starts.pop(event.node_index), event.time_s))
    for node, start in starts.items():
        intervals.setdefault(node, []).append((start, float(result.t[-1]) + np.finfo(float).eps))
    for node, node_intervals in intervals.items():
        for start, end in node_intervals:
            mask = (result.t >= start) & (result.t < end)
            short[mask, node] = parameters.spontaneous_power_w
    heater = np.zeros((n_time, n_nodes), dtype=float)
    if trigger.kind == "heater":
        stops = [event.time_s for event in result.events if event.name == "heater_stop"]
        stop = min(stops) if stops else float(result.t[-1]) + np.finfo(float).eps
        active = result.t < stop
        profile = trigger.heater_profile
        if profile is None:
            heater[np.ix_(active, parameters.trigger_nodes)] = trigger.heater_power_w_per_node
        else:
            total = profile.efficiency * np.interp(
                result.t[active],
                parameters.heater_profile_time_s,
                parameters.heater_profile_power_w,
                left=0.0,
                right=0.0,
            )
            heater[np.ix_(active, parameters.trigger_nodes)] = np.column_stack(
                (profile.front_fraction * total, (1.0 - profile.front_fraction) * total)
            )
    generated = reaction_total + short + heater
    net = generated + transfer
    temperature_rate = net / parameters.mass_cp[np.newaxis, :]
    return HeatPowerSeries(
        time_s=result.t.copy(),
        reaction_components_w=components,
        reaction_total_w=reaction_total,
        short_circuit_w=short,
        external_heater_w=heater,
        generated_w=generated,
        transfer_w=transfer,
        net_w=net,
        temperature_rate_k_per_s=temperature_rate,
        parameters=parameters,
    )


def extract_node_temperatures(states: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    return states[:, parameters.temperature_index :: parameters.states_per_node]


def extract_center_temperatures(states: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    temperatures = extract_node_temperatures(states, parameters)
    geometry = parameters.observation_geometry
    return 0.5 * (
        temperatures[:, geometry.battery_front_nodes]
        + temperatures[:, geometry.battery_back_nodes]
    )


def _battery_interface_temperatures(
    states: np.ndarray,
    parameters: ModelParameters,
) -> np.ndarray:
    temperatures = extract_node_temperatures(states, parameters)
    geometry = parameters.observation_geometry
    left = temperatures[:, geometry.interface_left_nodes]
    right = temperatures[:, geometry.interface_right_nodes]
    weights = geometry.interface_right_weights[None, :]
    return left + weights * (right - left)


def extract_face_temperatures(states: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    return _battery_interface_temperatures(states, parameters)


def extract_shell_surface_temperatures(
    states: np.ndarray,
    parameters: ModelParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return front/back metal-shell temperatures for every battery."""
    node_temperature = extract_node_temperatures(states, parameters)
    geometry = parameters.observation_geometry
    node_values = node_temperature[:, geometry.shell_nodes]
    outward = np.full_like(node_values, parameters.config.ambient_temperature_k)
    available = geometry.shell_neighbors >= 0
    for side in range(2):
        side_available = available[:, side]
        outward[:, side_available, side] = node_temperature[
            :, geometry.shell_neighbors[side_available, side]
        ]
    surface = node_values - geometry.shell_weights[None, :, :] * (node_values - outward)
    return surface[:, :, 0], surface[:, :, 1]


def calculate_all_interface_temperatures(
    states: np.ndarray, parameters: ModelParameters
) -> tuple[np.ndarray, np.ndarray]:
    n_time = states.shape[0]
    thermal = parameters.config.thermal
    battery = _battery_interface_temperatures(states, parameters)
    holder_columns: list[np.ndarray] = []
    r_cell = thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12
    r_holder = thermal.r_holder + thermal.r_ren
    if parameters.topology.has_holder_front:
        holder = parameters.node_map["holder_f"]
        cell = parameters.node_map["bat1_f"]
        t_holder = states[:, holder * parameters.states_per_node]
        t_cell = states[:, cell * parameters.states_per_node]
        holder_columns.append((t_holder * r_cell + t_cell * r_holder) / (r_holder + r_cell))
    if parameters.topology.has_holder_back:
        cell = parameters.node_map[f"bat{parameters.n_batteries}_b"]
        holder = parameters.node_map["holder_b"]
        t_cell = states[:, cell * parameters.states_per_node]
        t_holder = states[:, holder * parameters.states_per_node]
        holder_columns.append((t_cell * r_holder + t_holder * r_cell) / (r_cell + r_holder))
    holder = np.column_stack(holder_columns) if holder_columns else np.empty((n_time, 0), dtype=float)
    return battery, holder


def core_temperature_series(states: np.ndarray, parameters: ModelParameters) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_temperatures = extract_node_temperatures(states, parameters)
    geometry = parameters.observation_geometry
    front_node = node_temperatures[:, geometry.battery_front_nodes]
    back_node = node_temperatures[:, geometry.battery_back_nodes]
    previous = np.full_like(front_node, parameters.config.ambient_temperature_k)
    available = geometry.core_front_neighbors >= 0
    previous[:, available] = node_temperatures[:, geometry.core_front_neighbors[available]]
    front = front_node + geometry.core_front_weights[None, :] * (previous - front_node)
    back = back_node + geometry.core_back_weights[None, :] * (front_node - back_node)
    return front, back, 0.5 * (front + back)


@dataclass(frozen=True, slots=True)
class ResultSummary:
    global_peak_temperature_k: float
    global_peak_time_s: float
    cell_peak_temperature_k: list[float]
    cell_peak_time_s: list[float]
    cell_tr_time_s: list[float | None]
    propagation_interval_s: list[float | None]
    max_propagation_interval_s: float | None
    node_short_energy_j: list[float]
    cell_short_energy_j: list[float]
    max_short_energy_j: float
    elapsed_s: float
    segment_count: int
    time_points: int

    @property
    def peak_temperature_K(self) -> float:
        return self.global_peak_temperature_k

    @property
    def global_peak_temperature_K(self) -> float:
        return self.global_peak_temperature_k

    @property
    def cell_peak_temperature_K(self) -> list[float]:
        return self.cell_peak_temperature_k

    @property
    def cell_T_TR_ARC_time_s(self) -> list[float | None]:
        return self.cell_tr_time_s

    @property
    def node_short_energy_J(self) -> list[float]:
        return self.node_short_energy_j

    @property
    def cell_short_energy_J(self) -> list[float]:
        return self.cell_short_energy_j

    @property
    def max_short_energy_J(self) -> float:
        return self.max_short_energy_j

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_TR_result(result: SimulationResult) -> ResultSummary:
    parameters = result.params
    front, back, center = core_temperature_series(result.Y, parameters)
    flat_index = int(np.argmax(center))
    time_index, _cell_index = np.unravel_index(flat_index, center.shape)
    cell_peak_temperature = np.max(center, axis=0)
    peak_indices = np.argmax(center, axis=0)
    cell_peak_time = result.t[peak_indices]
    trigger = parameters.config.trigger
    if trigger.spontaneous_short_enabled:
        # Preserve the Feng/event definition whenever spontaneous short heat
        # is part of the configured physical model.
        tr_times: list[float | None] = [None] * parameters.n_batteries
        if trigger.kind == "needle":
            tr_times[trigger.cell - 1] = 0.0
        for event in result.events:
            if event.name != "spontaneous_short_start" or event.battery_index is None:
                continue
            cell = event.battery_index - 1
            if tr_times[cell] is None:
                tr_times[cell] = float(event.time_s)
    else:
        # Without the empirical short source, define TR from reaction
        # self-heating rather than from an event that cannot occur.
        tr_times = _kinetic_tr_times(result, trigger.kinetic_tr_rate_k_per_s)
        if trigger.kind == "needle":
            tr_times[trigger.cell - 1] = 0.0
    propagation = _adjacent_propagation_intervals(tr_times)
    finite_propagation = [value for value in propagation if value is not None]
    values = result.Y.reshape(result.Y.shape[0], parameters.n_nodes, parameters.states_per_node)
    node_energy = np.max(values[:, :, parameters.energy_index], axis=0)
    cell_energy = []
    for cell in range(parameters.n_batteries):
        f = parameters.node_map[f"bat{cell + 1}_f"]
        b = parameters.node_map[f"bat{cell + 1}_b"]
        cell_energy.append(float(node_energy[f] + node_energy[b]))
    return ResultSummary(
        global_peak_temperature_k=float(center[time_index, _cell_index]),
        global_peak_time_s=float(result.t[time_index]),
        cell_peak_temperature_k=cell_peak_temperature.astype(float).tolist(),
        cell_peak_time_s=cell_peak_time.astype(float).tolist(),
        cell_tr_time_s=tr_times,
        propagation_interval_s=propagation,
        max_propagation_interval_s=max(finite_propagation) if finite_propagation else None,
        node_short_energy_j=node_energy.astype(float).tolist(),
        cell_short_energy_j=cell_energy,
        max_short_energy_j=float(np.max(node_energy)),
        elapsed_s=float(result.elapsed_s),
        segment_count=len(result.segments),
        time_points=len(result.t),
    )


def _kinetic_tr_times(result: SimulationResult, threshold_k_per_s: float) -> list[float | None]:
    """Return first reaction-only self-heating-rate crossing for each cell."""
    parameters = result.params
    n_time = len(result.t)
    values = result.Y.reshape(n_time, parameters.n_nodes, parameters.states_per_node)
    reaction_heat = np.zeros((n_time, parameters.n_batteries), dtype=float)
    cell_mass_per_node, cell_cp = parameters.config.cell.effective()
    cell_heat_capacity = 2.0 * cell_mass_per_node * cell_cp
    temperatures = values[:, :, parameters.temperature_index]
    core_front, core_back, _center = core_temperature_series(result.Y, parameters)
    for battery_offset, node in enumerate(parameters.battery_nodes):
        node = int(node)
        cell = int(parameters.battery_cells[battery_offset])
        core = core_front[:, cell] if parameters.battery_is_front[battery_offset] else core_back[:, cell]
        _rates, heat_w = concentration_rates_batch(
            temperatures[:, node],
            core,
            values[:, node, parameters.concentration_slice],
            parameters.prepared_reactions,
        )
        reaction_heat[:, cell] += heat_w
    reaction_rate = reaction_heat / cell_heat_capacity
    return [
        _first_upward_crossing(result.t, reaction_rate[:, cell], threshold_k_per_s)
        for cell in range(parameters.n_batteries)
    ]


def _adjacent_propagation_intervals(
    tr_times: list[float | None],
) -> list[float | None]:
    """Return nonnegative delays aligned with physical adjacent-cell gaps."""
    propagation: list[float | None] = []
    for left, right in zip(tr_times[:-1], tr_times[1:]):
        # This is the delay across an adjacent-cell interface, not a signed
        # left-to-right quantity. Absolute time difference also supports a
        # heater located in the middle of an arbitrary-N module.
        propagation.append(float(abs(right - left)) if left is not None and right is not None else None)
    return propagation


def _first_upward_crossing(
    time_s: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float | None:
    """Linearly interpolate the first accepted-state upward crossing."""
    if len(time_s) == 0:
        return None
    if values[0] >= threshold:
        return float(time_s[0])
    indices = np.flatnonzero((values[:-1] < threshold) & (values[1:] >= threshold))
    if len(indices) == 0:
        return None
    left = int(indices[0])
    delta = float(values[left + 1] - values[left])
    if delta <= 0.0:
        return float(time_s[left + 1])
    fraction = float((threshold - values[left]) / delta)
    return float(time_s[left] + fraction * (time_s[left + 1] - time_s[left]))


def temperature_series_payload(result: SimulationResult) -> dict[str, Any]:
    front, back, center = core_temperature_series(result.Y, result.params)
    surface_front, surface_back = extract_shell_surface_temperatures(result.Y, result.params)
    return {
        "time_s": result.t.astype(float).tolist(),
        "front_k": front.T.astype(float).tolist(),
        "back_k": back.T.astype(float).tolist(),
        "center_k": center.T.astype(float).tolist(),
        "surface_front_k": surface_front.T.astype(float).tolist(),
        "surface_back_k": surface_back.T.astype(float).tolist(),
    }
