from __future__ import annotations

import numpy as np
import pytest

from dataclasses import replace

from moduletr import build_config
from moduletr.model import battery_rhs, initialize_parameters
from moduletr.postprocess import extract_shell_surface_temperatures
from moduletr.thermal import calculate_core_temperatures, calculate_heat_transfer, calculate_thermal_resistance, create_node_mapping


@pytest.mark.parametrize("holders, expected", [("none", 6), ("both", 8)])
def test_node_mapping(holders: str, expected: int) -> None:
    topology = create_node_mapping(3, holders)
    assert topology.n_nodes == expected
    assert topology.node_map["bat1_f"] == (1 if holders == "both" else 0)
    assert topology.node_map["bat3_b"] == expected - (2 if holders == "both" else 1)


def test_holder_states_keep_uniform_node_major_layout() -> None:
    parameters, state = initialize_parameters(
        build_config(n_batteries=5, holders="both", t_end=1)
    )
    values = state.reshape(parameters.n_nodes, parameters.states_per_node)
    derivative = battery_rhs(0.0, state, parameters).reshape(values.shape)
    assert parameters.states_per_node == parameters.n_reactions + 2
    assert values[parameters.topology.is_holder, parameters.concentration_slice] == pytest.approx(0.0)
    assert derivative[parameters.topology.is_holder, parameters.concentration_slice] == pytest.approx(0.0)


def test_uniform_temperature_has_zero_net_heat() -> None:
    parameters, _ = initialize_parameters(build_config(n_batteries=3, holders="both", t_end=1))
    temperature = np.full(parameters.n_nodes, parameters.config.ambient_temperature_k)
    assert calculate_heat_transfer(temperature, parameters) == pytest.approx(np.zeros(parameters.n_nodes), abs=1e-12)


def test_internal_conductance_is_energy_conservative() -> None:
    parameters, _ = initialize_parameters(build_config(n_batteries=2, holders="none", t_end=1))
    assert np.asarray(parameters.conductance.sum(axis=0)).ravel() == pytest.approx(np.zeros(parameters.n_nodes), abs=1e-12)


def test_core_temperature_literal_interpolation() -> None:
    parameters, _ = initialize_parameters(build_config(n_batteries=1, holders="none", t_end=1))
    temperatures = np.array([400.0, 500.0])
    front, back = calculate_core_temperatures(temperatures, parameters)
    thermal = parameters.config.thermal
    front_path = thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_convection
    back_path = 2.0 * (thermal.r_jr_12 + thermal.r_ap_in)
    assert front[0] == pytest.approx(400 + thermal.r_jr_12 / front_path * (299.15 - 400))
    assert back[0] == pytest.approx(500 + thermal.r_jr_12 / back_path * (400 - 500))


def test_adiabatic_boundary_removes_only_ambient_heat_transfer() -> None:
    config = build_config(n_batteries=2, holders="none", t_end=1)
    config = replace(config, thermal=replace(config.thermal, adiabatic_boundary=True))
    parameters, _ = initialize_parameters(config)
    assert parameters.ambient_coeff == pytest.approx(np.zeros(parameters.n_nodes))
    assert np.all(parameters.ambient_coeff_after_release > 0.0)
    assert parameters.conductance.nnz > 0


def test_all_resistance_paths_are_positive() -> None:
    parameters, _ = initialize_parameters(build_config(n_batteries=2, holders="both", t_end=1))
    for node in range(parameters.n_nodes):
        for direction in range(1, 7):
            assert calculate_thermal_resistance(node, direction, parameters.topology, parameters.config.thermal, 2) > 0


def test_shell_surface_observer_excludes_contact_and_barrier_resistance() -> None:
    config = build_config(n_batteries=2, holders="none", t_end=1, solver_preset="fast")
    parameters, state = initialize_parameters(config)
    values = state.reshape(parameters.n_nodes, parameters.states_per_node)
    left = parameters.node_map["bat1_b"]
    right = parameters.node_map["bat2_f"]
    values[left, 0] = 400.0
    values[right, 0] = 300.0
    front, back = extract_shell_surface_temperatures(values.ravel()[None, :], parameters)
    thermal = config.thermal
    total_raw = 2.0 * (
        thermal.r_contact + thermal.r_shell + thermal.r_ap_out + thermal.r_jr_12
    ) + thermal.r_add
    inner_raw = thermal.r_jr_12 + thermal.r_ap_out + thermal.r_shell
    expected_left = 400.0 - (400.0 - 300.0) * inner_raw / total_raw
    expected_right = 300.0 + (400.0 - 300.0) * inner_raw / total_raw
    assert back[0, 0] == pytest.approx(expected_left)
    assert front[0, 1] == pytest.approx(expected_right)
