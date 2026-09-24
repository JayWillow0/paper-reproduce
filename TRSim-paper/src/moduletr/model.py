from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from .config import ModuleConfig, validate_config
from .reactions import (
    PreparedReactionModel,
    apply_reaction_overrides,
    concentration_rates_batch,
    prepare_reaction_model,
)
from .thermal import (
    NodeTopology,
    ThermalObservationGeometry,
    assemble_thermal_network,
    calculate_core_temperatures,
    calculate_heat_transfer,
    create_node_mapping,
    prepare_observation_geometry,
)


@dataclass(slots=True)
class RuntimeModes:
    heater_active: bool
    trigger_short_active: bool
    spontaneous_short_active: np.ndarray
    spontaneous_short_done: np.ndarray
    adiabatic_boundary_active: bool


@dataclass(slots=True)
class ModelParameters:
    config: ModuleConfig
    topology: NodeTopology
    reactions: dict[str, dict[str, Any]]
    prepared_reactions: PreparedReactionModel
    reaction_names: tuple[str, ...]
    concentration_names: tuple[str, ...]
    n_reactions: int
    states_per_node: int
    temperature_index: int
    concentration_slice: slice
    energy_index: int
    mass_cp: np.ndarray
    conductance: csr_matrix
    ambient_coeff_initial: np.ndarray
    ambient_coeff_after_release: np.ndarray
    observation_geometry: ThermalObservationGeometry
    battery_nodes: np.ndarray
    battery_cells: np.ndarray
    battery_is_front: np.ndarray
    trigger_nodes: np.ndarray
    trigger_power_w: float
    spontaneous_power_w: float
    heater_profile_time_s: np.ndarray
    heater_profile_power_w: np.ndarray
    modes: RuntimeModes

    @property
    def n_nodes(self) -> int:
        return self.topology.n_nodes

    @property
    def n_batteries(self) -> int:
        return self.config.topology.n_cells

    @property
    def n_states_per_node(self) -> int:
        return self.states_per_node

    @property
    def node_map(self) -> dict[str, int]:
        return self.topology.node_map

    @property
    def model_type(self) -> str:
        return self.config.reactions.model

    @property
    def ambient_coeff(self) -> np.ndarray:
        if self.modes.adiabatic_boundary_active:
            return self.ambient_coeff_initial
        return self.ambient_coeff_after_release


def initialize_parameters(config: ModuleConfig) -> tuple[ModelParameters, np.ndarray]:
    config = validate_config(config)
    topology = create_node_mapping(config.topology.n_cells, config.topology.holders)
    reactions = apply_reaction_overrides(config.reactions.model, config.reactions.overrides)
    reaction_names = config.reactions.active
    concentration_names = reaction_names
    c0 = np.asarray([reactions[name]["c0"] for name in reaction_names], dtype=float)
    n_reactions = len(c0)
    states_per_node = n_reactions + 2
    energy_index = states_per_node - 1
    cell_mass, cell_cp = config.cell.effective()
    mass_cp = np.full(topology.n_nodes, cell_mass * cell_cp, dtype=float)
    mass_cp[topology.is_holder] = config.holder.mass_kg * config.holder.specific_heat_j_per_kg_k
    conductance, ambient_coeff = assemble_thermal_network(config, topology)
    nonadiabatic_config = replace(
        config,
        thermal=replace(
            config.thermal,
            adiabatic_boundary=False,
            adiabatic_release_temperature_k=None,
        ),
    )
    _, ambient_coeff_after_release = assemble_thermal_network(nonadiabatic_config, topology)
    ambient_coeff.setflags(write=False)
    ambient_coeff_after_release.setflags(write=False)
    observation_geometry = prepare_observation_geometry(config, topology)
    battery_nodes = np.flatnonzero(topology.is_battery)
    battery_nodes.setflags(write=False)
    battery_cells = topology.cell_index[battery_nodes].astype(int, copy=True)
    battery_is_front = topology.is_front[battery_nodes].astype(bool, copy=True)
    battery_cells.setflags(write=False)
    battery_is_front.setflags(write=False)
    trigger_nodes = np.asarray(
        [topology.node_map[f"bat{config.trigger.cell}_f"], topology.node_map[f"bat{config.trigger.cell}_b"]], dtype=int
    )
    modes = RuntimeModes(
        heater_active=config.trigger.kind == "heater",
        trigger_short_active=config.trigger.kind == "needle",
        spontaneous_short_active=np.zeros(topology.n_nodes, dtype=bool),
        spontaneous_short_done=np.zeros(topology.n_nodes, dtype=bool),
        adiabatic_boundary_active=config.thermal.adiabatic_boundary,
    )
    if config.trigger.kind == "needle":
        modes.spontaneous_short_done[trigger_nodes] = True
    heater_profile = config.trigger.heater_profile
    heater_profile_time_s = np.asarray(
        () if heater_profile is None else heater_profile.time_s,
        dtype=float,
    )
    heater_profile_power_w = np.asarray(
        () if heater_profile is None else heater_profile.total_power_w,
        dtype=float,
    )
    heater_profile_time_s.setflags(write=False)
    heater_profile_power_w.setflags(write=False)
    parameters = ModelParameters(
        config=config,
        topology=topology,
        reactions=reactions,
        prepared_reactions=prepare_reaction_model(
            config.reactions.model, reactions, reaction_names
        ),
        reaction_names=reaction_names,
        concentration_names=concentration_names,
        n_reactions=n_reactions,
        states_per_node=states_per_node,
        temperature_index=0,
        concentration_slice=slice(1, 1 + n_reactions),
        energy_index=energy_index,
        mass_cp=mass_cp,
        conductance=conductance,
        ambient_coeff_initial=ambient_coeff,
        ambient_coeff_after_release=ambient_coeff_after_release,
        observation_geometry=observation_geometry,
        battery_nodes=battery_nodes,
        battery_cells=battery_cells,
        battery_is_front=battery_is_front,
        trigger_nodes=trigger_nodes,
        trigger_power_w=config.trigger.trigger_energy_j / 2.0 / config.trigger.release_duration_s,
        spontaneous_power_w=config.trigger.spontaneous_energy_j / 2.0 / config.trigger.release_duration_s,
        heater_profile_time_s=heater_profile_time_s,
        heater_profile_power_w=heater_profile_power_w,
        modes=modes,
    )
    initial = np.zeros((topology.n_nodes, states_per_node), dtype=float)
    initial[:, 0] = config.ambient_temperature_k
    initial_cell_temperature = (
        config.ambient_temperature_k
        if config.initial_temperature_k is None
        else config.initial_temperature_k
    )
    initial[topology.is_battery, 0] = initial_cell_temperature
    initial[topology.is_battery, 1 : 1 + n_reactions] = c0
    return parameters, initial.ravel()


def battery_rhs(_time_s: float, state: np.ndarray, parameters: ModelParameters) -> np.ndarray:
    values = state.reshape(parameters.n_nodes, parameters.states_per_node)
    temperatures = values[:, 0]
    energy = values[:, parameters.energy_index]
    derivatives = np.zeros_like(values)
    core_front, core_back = calculate_core_temperatures(temperatures, parameters)
    reaction_heat = np.zeros(parameters.n_nodes, dtype=float)
    battery_core = np.where(
        parameters.battery_is_front,
        core_front[parameters.battery_cells],
        core_back[parameters.battery_cells],
    )
    rates, heat = concentration_rates_batch(
        temperatures[parameters.battery_nodes],
        battery_core,
        values[parameters.battery_nodes, parameters.concentration_slice],
        parameters.prepared_reactions,
    )
    derivatives[parameters.battery_nodes, parameters.concentration_slice] = rates
    reaction_heat[parameters.battery_nodes] = heat
    short_heat = np.zeros(parameters.n_nodes, dtype=float)
    if parameters.config.trigger.kind == "needle" and parameters.modes.trigger_short_active:
        for node in parameters.trigger_nodes:
            if parameters.config.trigger.trigger_energy_j / 2.0 - energy[node] > 0:
                short_heat[node] += parameters.trigger_power_w
                derivatives[node, parameters.energy_index] += parameters.trigger_power_w
    if parameters.config.trigger.kind == "heater" and parameters.modes.heater_active:
        short_heat[parameters.trigger_nodes] += heater_power_for_trigger_nodes(_time_s, parameters)
    for node in np.flatnonzero(parameters.modes.spontaneous_short_active & parameters.topology.is_battery):
        if parameters.config.trigger.spontaneous_energy_j / 2.0 - energy[node] > 0:
            short_heat[node] += parameters.spontaneous_power_w
            derivatives[node, parameters.energy_index] += parameters.spontaneous_power_w
    transfer_heat = calculate_heat_transfer(temperatures, parameters)
    derivatives[:, 0] = (reaction_heat + short_heat + transfer_heat) / parameters.mass_cp
    return derivatives.ravel()


def heater_power_for_trigger_nodes(time_s: float, parameters: ModelParameters) -> np.ndarray:
    """Return absorbed front/back heater power without reading or changing runtime history."""
    trigger = parameters.config.trigger
    profile = trigger.heater_profile
    if profile is None:
        return np.full(2, trigger.heater_power_w_per_node, dtype=float)
    total = profile.efficiency * float(
        np.interp(
            time_s,
            parameters.heater_profile_time_s,
            parameters.heater_profile_power_w,
            left=0.0,
            right=0.0,
        )
    )
    return np.asarray(
        [profile.front_fraction * total, (1.0 - profile.front_fraction) * total],
        dtype=float,
    )


def calculate_reaction_heat_for_node(state: np.ndarray, node: int, parameters: ModelParameters) -> float:
    values = state.reshape(parameters.n_nodes, parameters.states_per_node)
    temperatures = values[:, 0]
    core_front, core_back = calculate_core_temperatures(temperatures, parameters)
    cell = int(parameters.topology.cell_index[node])
    core_temperature = core_front[cell] if parameters.topology.is_front[node] else core_back[cell]
    _, heat = concentration_rates_batch(
        np.asarray([temperatures[node]], dtype=float),
        np.asarray([core_temperature], dtype=float),
        values[node, parameters.concentration_slice][None, :],
        parameters.prepared_reactions,
    )
    return float(heat[0])
