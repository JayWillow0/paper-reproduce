from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from moduletr import (
    ModuleConfig,
    ReactionConfig,
    TopologyConfig,
    TriggerConfig,
    build_config,
    reconstruct_heat_power,
    run_simulation,
    with_heater_profile,
)
from moduletr.model import heater_power_for_trigger_nodes, initialize_parameters
from moduletr.postprocess import _adjacent_propagation_intervals, _first_upward_crossing


def _fake_solution(t_span, y0, events, *, nonfinite: bool = False):
    t = np.asarray([t_span[0], t_span[1]], dtype=float)
    y = np.column_stack((y0, y0)).astype(float)
    if nonfinite:
        y[0, -1] = np.nan
    return SimpleNamespace(
        success=True,
        message="ok",
        t=t,
        y=y,
        t_events=[np.asarray([], dtype=float) for _event in events],
    )


def test_default_auto_short_smoke_and_json_payload() -> None:
    result = run_simulation(build_config(n_batteries=1, holders="none", t_end=2, solver_preset="fast"))
    assert result.t[0] == 0
    assert result.t[-1] == pytest.approx(2)
    assert np.all(np.isfinite(result.Y))
    assert all(segment.solver_method == "LSODA" for segment in result.segments)
    assert not any(event.name == "adiabatic_boundary_release" for event in result.events)
    json.dumps(result.to_dict("full"))


def test_finite_runtime_warning_is_recorded_without_solver_fallback(monkeypatch) -> None:
    methods: list[str] = []

    def fake_solve(_fun, t_span, y0, *, method, events, **_kwargs):
        methods.append(method)
        for _ in range(3):
            warnings.warn("recoverable trial", RuntimeWarning)
        return _fake_solution(t_span, y0, events)

    monkeypatch.setattr("moduletr.solver.solve_ivp", fake_solve)
    config = build_config(n_batteries=1, holders="none", t_end=1)
    config = replace(config, solver=replace(config.solver, method="BDF"))
    result = run_simulation(config)
    assert methods == ["BDF"]
    assert result.segments[0].solver_method == "BDF"
    assert result.segments[0].runtime_warnings == ("recoverable trial",)


def test_nonfinite_accepted_values_trigger_lsoda_fallback(monkeypatch) -> None:
    methods: list[str] = []

    def fake_solve(_fun, t_span, y0, *, method, events, **_kwargs):
        methods.append(method)
        return _fake_solution(t_span, y0, events, nonfinite=method == "BDF")

    monkeypatch.setattr("moduletr.solver.solve_ivp", fake_solve)
    config = build_config(n_batteries=1, holders="none", t_end=1)
    config = replace(config, solver=replace(config.solver, method="BDF"))
    result = run_simulation(config)
    assert methods == ["BDF", "LSODA"]
    assert result.segments[0].solver_method == "LSODA"


def test_kinetic_tr_crossing_is_linearly_interpolated() -> None:
    time_s = np.asarray([0.0, 2.0, 5.0, 9.0])
    rate = np.asarray([0.1, 0.6, 1.4, 2.0])
    assert _first_upward_crossing(time_s, rate, 1.0) == pytest.approx(3.5)
    assert _first_upward_crossing(time_s, rate, 3.0) is None


def test_adjacent_propagation_intervals_support_center_triggering() -> None:
    assert _adjacent_propagation_intervals([None, 7.0, 2.0, 8.0, None]) == [
        None,
        5.0,
        6.0,
        None,
    ]


def test_reaction_only_run_reports_kinetic_tr_without_short_event() -> None:
    config = ModuleConfig(
        topology=TopologyConfig(n_cells=1, holders="none"),
        reactions=ReactionConfig(model="standard"),
        trigger=TriggerConfig(
            kind="heater",
            heating_stop_mode="temperature",
            tr_threshold_k=2000.0,
            spontaneous_short_enabled=False,
            kinetic_tr_rate_k_per_s=1.0e-6,
        ),
        initial_temperature_k=500.0,
        t_end_s=0.1,
    )
    result = run_simulation(config)
    assert not any(event.name == "spontaneous_short_start" for event in result.events)
    # 从 500 K 热启动时，隔膜吸热反应在最初微秒内主导净热功率；待其浓度耗尽、
    # 放热反应占主导后，温升率上穿阈值。因此动理学 TR 时刻是一个很小的正数。
    tr_time = result.summary.cell_tr_time_s[0]
    assert tr_time is not None
    assert tr_time < 0.01


def test_initial_cell_temperature_is_independent_from_ambient_and_holder() -> None:
    config = ModuleConfig(
        topology=TopologyConfig(n_cells=2, holders="both"),
        ambient_temperature_k=298.15,
        initial_temperature_k=330.15,
        t_end_s=1.0,
    )
    parameters, state = initialize_parameters(config)
    values = state.reshape(parameters.n_nodes, parameters.states_per_node)
    assert values[parameters.topology.is_battery, 0] == pytest.approx(330.15)
    assert values[parameters.topology.is_holder, 0] == pytest.approx(298.15)


def test_initial_temperature_defaults_to_ambient_for_backward_compatibility() -> None:
    config = ModuleConfig(
        topology=TopologyConfig(n_cells=1, holders="both"),
        ambient_temperature_k=301.15,
        t_end_s=1.0,
    )
    parameters, state = initialize_parameters(config)
    values = state.reshape(parameters.n_nodes, parameters.states_per_node)
    assert values[:, 0] == pytest.approx(301.15)


def test_needle_event_and_energy_monotonicity() -> None:
    result = run_simulation(build_config(n_batteries=2, holders="none", t_end=30))
    assert any(event.name == "trigger_short_end" and event.time_s == pytest.approx(10) for event in result.events)
    values = result.Y.reshape(len(result.t), result.params.n_nodes, result.params.states_per_node)
    energy = values[:, :, result.params.energy_index]
    assert np.min(np.diff(energy, axis=0)) >= -1e-6
    assert np.min(energy) >= -1e-9


def test_heater_temperature_stop_and_simultaneous_start() -> None:
    config = build_config(n_batteries=1, holders="none", trigger_type=2, heating_stop_mode=1, t_end=10)
    config = replace(config, trigger=replace(config.trigger, tr_threshold_k=305.0))
    result = run_simulation(config)
    names = [event.name for event in result.events]
    assert "heater_stop" in names
    assert "spontaneous_short_start" in names
    assert "adiabatic_boundary_release" not in names


def test_spontaneous_short_can_be_disabled_without_disabling_trigger_short() -> None:
    config = build_config(n_batteries=2, holders="none", t_end=30, solver_preset="fast")
    config = replace(config, trigger=replace(config.trigger, spontaneous_short_enabled=False))
    result = run_simulation(config)
    assert any(event.name == "trigger_short_end" for event in result.events)
    assert not any(event.name.startswith("spontaneous_short") for event in result.events)
    assert result.summary.cell_tr_time_s == [0.0, None]
    assert result.summary.propagation_interval_s == [None]


def test_adiabatic_boundary_releases_at_configured_temperature() -> None:
    config = build_config(
        n_batteries=1,
        holders="none",
        trigger_type=2,
        t_end=5,
        solver_preset="fast",
    )
    config = replace(
        config,
        thermal=replace(
            config.thermal,
            adiabatic_boundary=True,
            adiabatic_release_temperature_k=300.0,
        ),
    )
    result = run_simulation(config)
    releases = [event for event in result.events if event.name == "adiabatic_boundary_release"]
    assert len(releases) == 1
    assert np.all(result.params.ambient_coeff > 0.0)
    heat = reconstruct_heat_power(result)
    transfer = heat.module_sum(heat.transfer_w, include_holders=True)
    assert transfer[result.t < releases[0].time_s] == pytest.approx(0.0, abs=1e-10)
    assert np.any(transfer[result.t >= releases[0].time_s] < 0.0)


def test_time_varying_heater_splits_total_power_and_stops_at_profile_end() -> None:
    config = ModuleConfig(
        topology=TopologyConfig(n_cells=3, holders="none"),
        t_end_s=8.0,
    )
    config = with_heater_profile(
        config,
        [0.0, 2.0, 5.0],
        [0.0, 1000.0, 400.0],
        cell=2,
        front_fraction=0.4,
        efficiency=0.8,
    )
    parameters, _state = initialize_parameters(config)
    assert heater_power_for_trigger_nodes(2.0, parameters) == pytest.approx([320.0, 480.0])
    assert heater_power_for_trigger_nodes(3.5, parameters) == pytest.approx([224.0, 336.0])
    result = run_simulation(config)
    stops = [event.time_s for event in result.events if event.name == "heater_stop"]
    assert stops == pytest.approx([5.0])
    heat = reconstruct_heat_power(result)
    total = heat.module_sum(heat.external_heater_w)
    before_stop = result.t < 5.0
    assert total[before_stop] == pytest.approx(
        0.8 * np.interp(result.t[before_stop], [0.0, 2.0, 5.0], [0.0, 1000.0, 400.0])
    )
    assert total[result.t >= 5.0] == pytest.approx(0.0)


def test_profile_heater_surface_temperature_interlock_forces_power_off() -> None:
    config = ModuleConfig(
        topology=TopologyConfig(n_cells=1, holders="none"),
        t_end_s=10.0,
    )
    config = with_heater_profile(
        config,
        [0.0, 10.0],
        [5000.0, 5000.0],
        cell=1,
        front_fraction=0.5,
        efficiency=1.0,
    )
    config = replace(
        config,
        trigger=replace(
            config.trigger,
            heater_stop_surface_temperature_k=config.ambient_temperature_k + 0.2,
            spontaneous_short_enabled=False,
        ),
    )
    result = run_simulation(config)
    stops = [event.time_s for event in result.events if event.name == "heater_stop"]
    assert len(stops) == 1
    assert 0.0 < stops[0] < 10.0
    heat = reconstruct_heat_power(result)
    total = heat.module_sum(heat.external_heater_w)
    assert total[result.t < stops[0]] == pytest.approx(5000.0)
    assert total[result.t >= stops[0]] == pytest.approx(0.0)


def test_profile_heater_surface_rate_interlock_can_stop_at_start() -> None:
    config = ModuleConfig(
        topology=TopologyConfig(n_cells=1, holders="none"),
        t_end_s=10.0,
    )
    config = with_heater_profile(
        config,
        [0.0, 10.0],
        [5000.0, 5000.0],
        cell=1,
        front_fraction=0.5,
        efficiency=1.0,
    )
    config = replace(
        config,
        trigger=replace(
            config.trigger,
            heater_stop_surface_rate_k_per_s=0.01,
            spontaneous_short_enabled=False,
        ),
    )
    result = run_simulation(config)
    stops = [event.time_s for event in result.events if event.name == "heater_stop"]
    assert stops == pytest.approx([0.0])
    heat = reconstruct_heat_power(result)
    assert heat.module_sum(heat.external_heater_w) == pytest.approx(0.0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "max_step, baseline_temperature, baseline_energy",
    [(1.0, 1509.882892, 199977.340454), (0.5, 1509.975927, 199989.765741), (0.1, 1510.371857, 199992.575788)],
)
def test_corrected_core_temperature_regression_band(max_step: float, baseline_temperature: float, baseline_energy: float) -> None:
    config = build_config(
        n_batteries=2,
        holders="none",
        trigger_type=1,
        trigger_battery=1,
        t_end=80,
        solver_options={"MaxStep": max_step, "RelTol": 7e-4},
    )
    # Pin LSODA and a slightly tighter tolerance so the corrected Feng
    # Eq. (28)/(29) event semantics remain reproducible across refactors.
    config = replace(config, solver=replace(config.solver, method="LSODA"))
    result = run_simulation(config)
    values = result.Y.reshape(len(result.t), result.params.n_nodes, result.params.states_per_node)
    peak = float(np.max(values[:, :, 0]))
    assert abs(peak - baseline_temperature) <= 2.0
    assert abs(result.summary.max_short_energy_j - baseline_energy) <= 200.0
