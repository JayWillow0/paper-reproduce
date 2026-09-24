from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from moduletr import (
    CalibrationError,
    FitParameter,
    SensitivityParameter,
    SimulationCase,
    build_config,
    calculate_all_interface_temperatures,
    extract_center_temperatures,
    extract_shell_surface_temperatures,
    plot_all_interface_temperatures,
    plot_concentrations,
    plot_temperatures,
    reconstruct_heat_power,
    run_TR_calibration,
    run_TR_sensitivity,
    run_simulation,
    run_simulation_sweep,
)


def test_calibration_and_sensitivity_smoke() -> None:
    config = build_config(n_batteries=1, holders="none", t_end=1, solver_preset="fast")
    result = run_simulation(config)
    target = {"time_s": result.t, "T_cell1_center_K": result.to_dict("series")["temperature_series"]["center_k"][0]}
    calibration = run_TR_calibration(
        config,
        target,
        [FitParameter("E_short_total", 4e5, 3.9e5, 4.1e5, scale=1e5)],
        {"algorithm": "fminsearch", "max_iter": 1, "random_starts": 1, "verbose": False},
    )
    assert np.isfinite(calibration.best_score)
    sensitivity = run_TR_sensitivity(
        config,
        [SensitivityParameter("E_short_total", (3.9e5, 4.1e5), "J")],
        {"mode": "oat", "verbose": False},
    )
    assert len(sensitivity) == 2


def test_calibration_rejects_invalid_target_before_optimization() -> None:
    config = build_config(n_batteries=1, holders="none", t_end=0.1, solver_preset="fast")
    with pytest.raises(ValueError, match="Unsupported target signal"):
        run_TR_calibration(
            config,
            {"time_s": [0.0, 0.1], "bad_signal": [299.15, 299.15]},
            [FitParameter("ambient_temperature_k", 299.15, 298.0, 300.0)],
            {"algorithm": "differential_evolution", "max_iter": 0, "population_size": 1},
        )


def test_calibration_failure_history_and_all_failed_error() -> None:
    config = build_config(n_batteries=1, holders="none", t_end=0.1, solver_preset="fast")
    result = run_simulation(config)
    target = {
        "time_s": result.t,
        "T_cell1_center_K": result.to_dict("series")["temperature_series"]["center_k"][0],
    }
    with pytest.raises(CalibrationError, match="All calibration evaluations failed"):
        run_TR_calibration(
            config,
            target,
            [FitParameter("trigger.cell", 2.0, 1.5, 2.5)],
            {
                "algorithm": "differential_evolution",
                "max_iter": 0,
                "population_size": 1,
                "rng_seed": 2,
            },
        )

    calibration = run_TR_calibration(
        config,
        target,
        [FitParameter("thermal.r_add", 0.1, -1.0, 1.0)],
        {
            "algorithm": "differential_evolution",
            "max_iter": 0,
            "population_size": 5,
            "rng_seed": 3,
        },
    )
    assert {record["status"] for record in calibration.history} == {"ok", "failed"}
    failed = next(record for record in calibration.history if record["status"] == "failed")
    assert failed["error_type"] == "ValueError"
    assert failed["error_message"]


@pytest.mark.slow
def test_explicit_parallel_workflows_preserve_order() -> None:
    config = build_config(n_batteries=1, holders="none", t_end=0.2, solver_preset="fast")
    cases = [
        SimulationCase("first", config),
        SimulationCase("second", config),
    ]
    sweep = run_simulation_sweep(cases, workers=2)
    assert [run.case.label for run in sweep] == ["first", "second"]

    sensitivity = run_TR_sensitivity(
        config,
        [SensitivityParameter("E_short_total", (3.9e5, 4.1e5), "J")],
        {"mode": "oat", "verbose": False, "workers": 2},
    )
    assert [record["run_index"] for record in sensitivity] == [1, 2]

    result = run_simulation(config)
    target = {
        "time_s": result.t,
        "T_cell1_center_K": result.to_dict("series")["temperature_series"]["center_k"][0],
    }
    calibration = run_TR_calibration(
        config,
        target,
        [FitParameter("ambient_temperature_k", 299.15, 298.0, 300.0)],
        {
            "algorithm": "differential_evolution",
            "max_iter": 0,
            "population_size": 5,
            "rng_seed": 4,
            "workers": 2,
        },
    )
    assert calibration.best_score >= 0.0
    assert all(record["status"] == "ok" for record in calibration.history)


def test_plot_exports_have_expected_pixel_sizes(tmp_path: Path) -> None:
    config = build_config(n_batteries=2, holders="both", t_end=1, solver_preset="fast")
    result = run_simulation(config)
    center = extract_center_temperatures(result.Y, result.params)
    battery_interface, holder_interface = calculate_all_interface_temperatures(result.Y, result.params)
    temp_path = tmp_path / "temp.png"
    interface_path = tmp_path / "temp_inter.png"
    concentration_path = tmp_path / "conc.png"
    figure_a = plot_temperatures(result.t, center, battery_interface, {"export_path": temp_path})
    figure_b = plot_all_interface_temperatures(result.t, battery_interface, holder_interface, result.params, {"export_path": interface_path})
    node = result.params.node_map["bat2_b"]
    figure_c = plot_concentrations(result.t, result.Y, result.params, node, {"export_path": concentration_path})
    import matplotlib.pyplot as plt
    from PIL import Image

    plt.close(figure_a)
    plt.close(figure_b)
    plt.close(figure_c)
    assert Image.open(temp_path).size == (1393, 946)
    assert Image.open(interface_path).size == (1392, 945)
    assert Image.open(concentration_path).size == (1608, 981)


def test_heat_reconstruction_and_full_sweep_retain_series() -> None:
    standard = build_config(n_batteries=1, holders="none", t_end=2, solver_preset="fast")
    result = run_simulation(standard)
    heat = reconstruct_heat_power(result)
    component_sum = np.sum(np.stack(tuple(heat.reaction_components_w.values())), axis=0)
    assert component_sum == pytest.approx(heat.reaction_total_w)
    assert heat.generated_w == pytest.approx(heat.reaction_total_w + heat.short_circuit_w + heat.external_heater_w)
    assert heat.net_w == pytest.approx(heat.generated_w + heat.transfer_w)
    runs = run_simulation_sweep([SimulationCase("standard", standard, "case", 1.0, "-")])
    assert runs[0].result.Y.shape == result.Y.shape
    assert runs[0].metrics["cell_tr_time_s"] == [0.0]


def test_calibration_accepts_shell_surface_targets_with_missing_samples() -> None:
    config = build_config(n_batteries=1, holders="none", t_end=1, solver_preset="fast")
    result = run_simulation(config)
    surface_front, _surface_back = extract_shell_surface_temperatures(result.Y, result.params)
    target_values = surface_front[:, 0].copy()
    if len(target_values) > 2:
        target_values[1] = np.nan
    calibration = run_TR_calibration(
        config,
        {"time_s": result.t, "T_cell1_surface_front_K": target_values},
        [
            FitParameter(
                "thermal.r_convection",
                config.thermal.r_convection,
                0.03,
                0.05,
                scale=0.04,
            )
        ],
        {
            "algorithm": "fminsearch",
            "max_iter": 1,
            "random_starts": 1,
            "verbose": False,
        },
    )
    assert np.isfinite(calibration.best_score)
