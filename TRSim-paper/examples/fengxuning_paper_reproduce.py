from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
from pathlib import Path
import re
from typing import Callable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from moduletr import (
    ModuleConfig,
    ReactionConfig,
    SimulationCase,
    SolverConfig,
    TopologyConfig,
    TriggerConfig,
    calculate_all_interface_temperatures,
    extract_center_temperatures,
    reconstruct_heat_power,
    run_simulation,
    run_simulation_sweep,
    with_heat_dissipation,
    with_short_energy_scale,
    with_thermal_barrier,
    without_thermal_barrier,
)
from moduletr.postprocess import core_temperature_series


STANDARD_REACTIONS = ("SEI", "anode", "separator", "electrolyte", "cathode1", "cathode2")
CELL_COLORS = (
    "#4C78A8",
    "#E39C53",
    "#72A276",
    "#C96B6B",
    "#8172A2",
    "#9A7660",
)
REACTION_COLORS = (
    "#4C78A8",
    "#72A276",
    "#C96B6B",
    "#8172A2",
    "#5F9EA0",
    "#9A7660",
)
STUDY_COLORS = (
    "#465872",
    "#5E718D",
    "#768AA5",
    "#8FA3BA",
    "#A8BCCF",
    "#C1D3DF",
    "#88768F",
    "#A38EA5",
)


def apply_publication_style() -> None:
    """Apply a compact, editable, low-saturation publication style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
        }
    )


apply_publication_style()


def parse_figures(value: str) -> list[int]:
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    invalid = sorted(selected - set(range(10, 20)))
    if invalid or not selected:
        raise argparse.ArgumentTypeError("figures must select one or more figures in [10, 19].")
    return sorted(selected)


def base_config(preset: str, *, n_cells: int = 6, t_end_s: float = 2000.0) -> ModuleConfig:
    return ModuleConfig(
        topology=TopologyConfig(n_cells=n_cells, holders="both"),
        trigger=TriggerConfig(kind="needle", cell=1),
        reactions=ReactionConfig(model="standard", active=STANDARD_REACTIONS),
        solver=SolverConfig(method="auto", preset=preset),
        t_end_s=t_end_s,
        ambient_temperature_k=299.15,
    ).normalized()


def finish_figure(figure: plt.Figure, number: int, output_dir: Path, dpi: int) -> None:
    figure.savefig(output_dir / f"fig{number}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / f"fig{number}.svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def style_axis(axis: plt.Axes, label: str) -> None:
    axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontsize=9, fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(direction="out")


def save_single_npz(output_dir: Path, number: int, result, **arrays: np.ndarray) -> None:
    np.savez_compressed(
        output_dir / f"fig{number}_data.npz",
        time_s=result.t,
        state=result.Y,
        node_labels=np.asarray(result.params.topology.labels, dtype=str),
        **arrays,
    )


def cell_heat_series(heat, values: np.ndarray, cell: int = 1) -> np.ndarray:
    return heat.cell_sum(values, cell)


def plot_fig10(preset: str, output_dir: Path, dpi: int) -> None:
    config = base_config(preset, n_cells=1, t_end_s=190_000.0)
    config = replace(
        config,
        topology=TopologyConfig(n_cells=1, holders="none"),
        ambient_temperature_k=323.15,
        thermal=replace(config.thermal, adiabatic_boundary=True),
        trigger=replace(
            config.trigger,
            kind="heater",
            trigger_energy_j=317_207.0,
            spontaneous_energy_j=317_207.0,
            heater_power_w_per_node=0.5,
            heating_stop_mode="self_heating_rate",
        ),
    )
    scout = run_simulation(config)
    scout_center_k = extract_center_temperatures(scout.Y, scout.params)[:, 0]
    release_temperature_k = max(
        config.ambient_temperature_k + 0.01,
        float(np.max(scout_center_k)) - 0.5,
    )
    config = replace(
        config,
        thermal=replace(
            config.thermal,
            adiabatic_release_temperature_k=release_temperature_k,
        ),
    )
    result = run_simulation(config)
    heat = reconstruct_heat_power(result)
    center_c = extract_center_temperatures(result.Y, result.params)[:, 0] - 273.15
    battery_nodes = np.flatnonzero(result.params.topology.is_battery)
    dtdt_c_per_min = np.mean(heat.temperature_rate_k_per_s[:, battery_nodes], axis=1) * 60.0
    generated = cell_heat_series(heat, heat.generated_w)
    short = cell_heat_series(heat, heat.short_circuit_w)
    components = {name: cell_heat_series(heat, values) for name, values in heat.reaction_components_w.items()}
    release_time_s = next(
        (event.time_s for event in result.events if event.name == "adiabatic_boundary_release"),
        np.inf,
    )
    heating_mask = result.t <= release_time_s + np.finfo(float).eps
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), constrained_layout=True)
    axes[0, 0].plot(result.t / 60.0, center_c, color="black", label="Simulation")
    axes[0, 0].set(xlabel="Time (min)", ylabel="Temperature (°C)")
    rate = np.clip(np.abs(dtdt_c_per_min), 1e-6, None)
    axes[0, 1].semilogy(center_c[heating_mask], rate[heating_mask], color="black")
    axes[0, 1].set(xlabel="Temperature (°C)", ylabel="Temperature rate (°C/min)")
    mask = heating_mask & (center_c <= min(320.0, float(np.max(center_c))))
    axes[1, 0].semilogy(center_c[mask], rate[mask], color="black")
    axes[1, 0].set(xlabel="Temperature (°C)", ylabel="Temperature rate (°C/min)")
    axes[1, 1].semilogy(center_c[heating_mask], np.clip(np.abs(generated[heating_mask]), 1e-6, None), color="black", linewidth=2, label="Q generated")
    for index, (name, values) in enumerate(components.items()):
        axes[1, 1].semilogy(center_c[heating_mask], np.clip(np.abs(values[heating_mask]), 1e-6, None), label=f"Q {name}", color=REACTION_COLORS[index % len(REACTION_COLORS)])
    axes[1, 1].semilogy(center_c[heating_mask], np.clip(np.abs(short[heating_mask]), 1e-6, None), label="Q short", color="#5F9EA0", linewidth=1.5)
    axes[1, 1].set(xlabel="Temperature (°C)", ylabel="Heat generation power (W)")
    axes[1, 1].legend(fontsize=6, ncol=2, loc="lower right")
    for label, axis in zip("abcd", axes.flat, strict=True):
        style_axis(axis, label)
    finish_figure(figure, 10, output_dir, dpi)
    save_single_npz(
        output_dir,
        10,
        result,
        center_temperature_c=center_c,
        temperature_rate_c_per_min=dtdt_c_per_min,
        generated_w=generated,
        short_circuit_w=short,
        adiabatic_release_temperature_k=np.asarray(release_temperature_k),
        adiabatic_release_time_s=np.asarray(release_time_s if np.isfinite(release_time_s) else np.nan),
        **{f"reaction_{name}_w": values for name, values in components.items()},
    )


def plot_fig11(preset: str, output_dir: Path, dpi: int) -> None:
    config = base_config(preset, n_cells=1, t_end_s=400.0)
    config = replace(
        config,
        topology=TopologyConfig(n_cells=1, holders="none"),
        ambient_temperature_k=298.15,
        trigger=replace(config.trigger, trigger_energy_j=380_000.0, release_duration_s=5.0),
    )
    result = run_simulation(config)
    heat = reconstruct_heat_power(result)
    center_c = extract_center_temperatures(result.Y, result.params)[:, 0] - 273.15
    generated = cell_heat_series(heat, heat.generated_w)
    short = cell_heat_series(heat, heat.short_circuit_w)
    components = {name: cell_heat_series(heat, values) for name, values in heat.reaction_components_w.items()}
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(result.t, center_c, color="black")
    axes[0].set(xlabel="Time (s)", ylabel="Temperature (°C)")
    axes[1].semilogy(result.t, np.clip(np.abs(generated), 1e-6, None), color="black", linewidth=2, label="Q generated")
    for index, (name, values) in enumerate(components.items()):
        axes[1].semilogy(result.t, np.clip(np.abs(values), 1e-6, None), label=f"Q {name}", color=REACTION_COLORS[index % len(REACTION_COLORS)])
    axes[1].semilogy(result.t, np.clip(np.abs(short), 1e-6, None), label="Q short", color="#5F9EA0")
    axes[1].set(xlabel="Time (s)", ylabel="Heat generation power (W)", xlim=(0, min(20, result.t[-1])))
    axes[1].legend(fontsize=7, ncol=2)
    for label, axis in zip("ab", axes, strict=True):
        style_axis(axis, label)
    finish_figure(figure, 11, output_dir, dpi)
    save_single_npz(output_dir, 11, result, center_temperature_c=center_c, generated_w=generated, short_circuit_w=short, **{f"reaction_{name}_w": values for name, values in components.items()})


def module_result(preset: str, *, t_end_s: float = 2000.0, ambient_c: float = 26.0, trigger_energy_j: float = 400_000.0):
    config = base_config(preset, n_cells=6, t_end_s=t_end_s)
    config = replace(
        config,
        ambient_temperature_k=ambient_c + 273.15,
        trigger=replace(config.trigger, trigger_energy_j=trigger_energy_j, spontaneous_energy_j=370_000.0, release_duration_s=10.0),
    )
    return run_simulation(config)


def plot_fig12(preset: str, output_dir: Path, dpi: int) -> None:
    result = module_result(preset)
    center_c = extract_center_temperatures(result.Y, result.params) - 273.15
    interface, _ = calculate_all_interface_temperatures(result.Y, result.params)
    interface_c = interface - 273.15
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, constrained_layout=True)
    for cell in range(center_c.shape[1]):
        axes[0].plot(result.t, center_c[:, cell], color=CELL_COLORS[cell], label=f"T{cell + 1}")
    for index in range(interface_c.shape[1]):
        axes[1].plot(result.t, interface_c[:, index], color=CELL_COLORS[index], label=f"T{index + 1},{index + 2}")
    axes[0].set(ylabel="Battery temperature (°C)")
    axes[1].set(xlabel="Time (s)", ylabel="Interface temperature (°C)")
    axes[1].set_xlim(0.0, 500.0)
    axes[0].legend(ncol=3, fontsize=8)
    axes[1].legend(ncol=3, fontsize=8)
    for label, axis in zip("ab", axes, strict=True):
        style_axis(axis, label)
    finish_figure(figure, 12, output_dir, dpi)
    save_single_npz(output_dir, 12, result, center_temperature_c=center_c, interface_temperature_c=interface_c)


def plot_fig13(preset: str, output_dir: Path, dpi: int) -> None:
    result = module_result(preset)
    front, back, _ = core_temperature_series(result.Y, result.params)
    figure = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 1, height_ratios=(1.0, 0.15))
    axis = figure.add_subplot(grid[0, 0])
    legend_axis = figure.add_subplot(grid[1, 0])
    for cell in range(result.params.n_batteries):
        axis.plot(result.t, front[:, cell] - 273.15, color=CELL_COLORS[cell], label=f"T~{cell + 1}_f")
        axis.plot(result.t, back[:, cell] - 273.15, color=CELL_COLORS[cell], linestyle="--", label=f"T~{cell + 1}_b")
    axis.axhline(result.params.config.trigger.tr_threshold_k - 273.15, color="black", linestyle=":", label="TTR,ARC")
    axis.set(xlabel="Time (s)", ylabel="Core-edge temperature (°C)")
    axis.set_xlim(0.0, 500.0)
    handles, labels = axis.get_legend_handles_labels()
    legend_axis.legend(handles, labels, ncol=5, loc="center", fontsize=6.2)
    legend_axis.set_axis_off()
    style_axis(axis, "a")
    finish_figure(figure, 13, output_dir, dpi)
    save_single_npz(output_dir, 13, result, core_front_c=front - 273.15, core_back_c=back - 273.15)


def plot_fig14(preset: str, output_dir: Path, dpi: int) -> None:
    normal = module_result(preset, t_end_s=1000.0)
    disabled_config = replace(normal.params.config, trigger=replace(normal.params.config.trigger, spontaneous_short_enabled=False))
    disabled = run_simulation(disabled_config)
    figure, axis = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    data: dict[str, np.ndarray] = {}
    for result, style, suffix in ((normal, "-", "normal"), (disabled, "--", "disabled")):
        center = extract_center_temperatures(result.Y, result.params) - 273.15
        front, back, _ = core_temperature_series(result.Y, result.params)
        axis.plot(result.t, center[:, 0], color=CELL_COLORS[0], linestyle=style, label=f"T1 {suffix}")
        axis.plot(result.t, center[:, 1], color=CELL_COLORS[1], linestyle=style, label=f"T2 {suffix}")
        axis.plot(result.t, front[:, 1] - 273.15, color=REACTION_COLORS[2], linestyle=style, label=f"T~2_f {suffix}")
        axis.plot(result.t, back[:, 1] - 273.15, color=REACTION_COLORS[3], linestyle=style, label=f"T~2_b {suffix}")
        data[f"{suffix}_time_s"] = result.t
        data[f"{suffix}_center_c"] = center
        data[f"{suffix}_core2_front_c"] = front[:, 1] - 273.15
        data[f"{suffix}_core2_back_c"] = back[:, 1] - 273.15
    axis.axhline(260.0, color="black", linestyle=":", label="TTR,ARC")
    axis.set(xlabel="Time (s)", ylabel="Temperature (°C)")
    axis.legend(ncol=2, fontsize=7)
    style_axis(axis, "a")
    finish_figure(figure, 14, output_dir, dpi)
    np.savez_compressed(output_dir / "fig14_data.npz", **data)


def export_sweep_data(number: int, runs, output_dir: Path) -> None:
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for index, run in enumerate(runs):
        key = f"case_{index:02d}_{re.sub(r'[^A-Za-z0-9]+', '_', run.case.label).strip('_')}"
        center = extract_center_temperatures(run.result.Y, run.result.params) - 273.15
        front, back, _ = core_temperature_series(run.result.Y, run.result.params)
        arrays[f"{key}_time_s"] = run.result.t
        arrays[f"{key}_center_c"] = center
        arrays[f"{key}_core2_max_c"] = np.maximum(front[:, 1], back[:, 1]) - 273.15
        rows.append(run.to_record())
    np.savez_compressed(output_dir / f"fig{number}_curves.npz", **arrays)
    with (output_dir / f"fig{number}_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fieldnames = list(rows[0])
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ";".join("" if value is None else str(value) for value in item) if isinstance(item, list) else item for key, item in row.items()})


def plot_parameter_study(
    number: int,
    runs,
    output_dir: Path,
    dpi: int,
    *,
    reverse_x: bool = False,
    reference_temperature_c: float = 260.0,
) -> None:
    figure = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    outer = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0), height_ratios=(1.15, 1.0))
    panel_a_grid = outer[0, 0].subgridspec(2, 2, hspace=0.08, wspace=0.10)
    panel_a_axes = [figure.add_subplot(panel_a_grid[row, column]) for row in range(2) for column in range(2)]
    axis_b = figure.add_subplot(outer[0, 1])
    axis_c = figure.add_subplot(outer[1, 0])
    axis_d = figure.add_subplot(outer[1, 1])
    representative = np.linspace(0, len(runs) - 1, min(4, len(runs)), dtype=int)
    time_limit_s = 1000.0 if number == 15 else 500.0
    for representative_axis, run_index in zip(panel_a_axes, representative, strict=True):
        run = runs[int(run_index)]
        center = extract_center_temperatures(run.result.Y, run.result.params) - 273.15
        for cell in range(center.shape[1]):
            representative_axis.plot(
                run.result.t,
                center[:, cell],
                color=CELL_COLORS[cell],
                label=f"T{cell + 1}",
            )
        representative_axis.set_xlim(0.0, time_limit_s)
        representative_axis.text(
            0.04,
            0.94,
            run.case.label,
            transform=representative_axis.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#3F3F3F",
        )
        representative_axis.spines[["top", "right"]].set_visible(False)
    for axis in panel_a_axes[:2]:
        axis.tick_params(labelbottom=False)
    for axis in panel_a_axes[1::2]:
        axis.tick_params(labelleft=False)
    panel_a_axes[2].set(xlabel="Time (s)", ylabel="Temperature (°C)")
    panel_a_axes[0].legend(ncol=2, loc="center right", fontsize=5.2, handlelength=1.3)
    style_axis(panel_a_axes[0], "a")
    transitions = np.arange(1, runs[0].result.params.n_batteries)
    for run_index, run in enumerate(runs):
        color = STUDY_COLORS[run_index % len(STUDY_COLORS)]
        intervals = np.asarray([np.nan if value is None else value for value in run.result.summary.propagation_interval_s])
        axis_b.plot(transitions, intervals, marker="o", markersize=3.2, color=color, label=run.case.label)
        front, back, _ = core_temperature_series(run.result.Y, run.result.params)
        axis_c.plot(
            run.result.t,
            np.maximum(front[:, 1], back[:, 1]) - 273.15,
            color=color,
            label=run.case.label,
        )
    has_propagation = any(
        any(value is not None for value in run.result.summary.propagation_interval_s)
        for run in runs
    )
    axis_b.set(xlabel="Propagation interval i→i+1", ylabel="Duration (s)", xticks=transitions)
    if not has_propagation:
        axis_b.text(0.5, 0.5, "No propagation", transform=axis_b.transAxes, ha="center", va="center", color="0.35")
        axis_b.set(xlim=(0.5, len(transitions) + 0.5), ylim=(0.0, 1.0))
    axis_c.axhline(reference_temperature_c, color="#555555", linestyle=":")
    axis_c.set(xlabel="Time (s)", ylabel="max(T~2_f, T~2_b) (°C)", xlim=(0.0, time_limit_s))
    x_values = np.asarray([float(run.case.parameter_value) for run in runs])
    d12 = np.asarray([np.nan if not run.result.summary.propagation_interval_s or run.result.summary.propagation_interval_s[0] is None else run.result.summary.propagation_interval_s[0] for run in runs])
    finite_x = np.isfinite(x_values)
    display_x = x_values.copy()
    if not np.all(finite_x):
        finite_max = float(np.max(x_values[finite_x])) if np.any(finite_x) else 1.0
        display_x[~finite_x] = finite_max * 10.0
    order = np.argsort(display_x)
    if np.any(np.isfinite(d12)):
        axis_d.plot(display_x[order], d12[order], marker="s", markersize=3.5, color="#4C78A8")
    else:
        axis_d.text(0.5, 0.5, "No Bat1→Bat2 propagation", transform=axis_d.transAxes, ha="center", va="center", color="0.35")
        axis_d.set(xlim=(float(np.min(display_x)), float(np.max(display_x))), ylim=(0.0, 1.0))
    axis_d.set(xlabel=f"{runs[0].case.parameter_name} ({runs[0].case.unit})", ylabel="D1→2 (s)")
    if not np.all(finite_x):
        axis_d.set_xscale("log")
        tick_mask = (~finite_x) | np.isclose(x_values, 10.0) | np.isclose(x_values, 1.0) | np.isclose(x_values, 0.2) | np.isclose(x_values, 0.04)
        tick_indices = np.flatnonzero(tick_mask)
        tick_indices = tick_indices[np.argsort(display_x[tick_indices])]
        axis_d.set_xticks(display_x[tick_indices])
        axis_d.set_xticklabels(
            ["∞" if not np.isfinite(x_values[index]) else f"{x_values[index]:g}" for index in tick_indices]
        )
    if reverse_x:
        axis_d.invert_xaxis()
    if has_propagation:
        axis_b.legend(fontsize=5.5, ncol=2)
    axis_c.legend(fontsize=5.5, ncol=2)
    for label, axis in zip("bcd", (axis_b, axis_c, axis_d), strict=True):
        style_axis(axis, label)
    finish_figure(figure, number, output_dir, dpi)
    export_sweep_data(number, runs, output_dir)


def make_cases(preset: str, number: int) -> list[SimulationCase]:
    base = base_config(preset, n_cells=6, t_end_s=2000.0)
    if number == 15:
        values = (170.0, 260.0, 360.0, 470.0, 560.0, 770.0)
        return [SimulationCase(f"{value:g}°C", replace(base, trigger=replace(base.trigger, tr_threshold_k=value + 273.15)), "TTR,ARC", value, "°C") for value in values]
    if number == 16:
        values = (1.20, 1.10, 1.00, 0.90, 0.65, 0.4)
        return [SimulationCase(f"{100 * value:g}% ΔHe", with_short_energy_scale(base, value), "ΔHe", 100 * value, "%") for value in values]
    if number == 17:
        values = (5.0, 10.0, 25.0, 50.0, 80.0, 100.0, 200.0)
        return [SimulationCase(f"hdis={value:g}", with_heat_dissipation(base, value), "hdis", value, "W m-2 K-1") for value in values]
    if number == 18:
        values = (math.inf, 10.0, 1.0, 0.6, 0.3, 0.2, 0.08, 0.04)
        output = []
        for value in values:
            config = without_thermal_barrier(base) if math.isinf(value) else with_thermal_barrier(base, 0.001, value)
            output.append(SimulationCase("no layer" if math.isinf(value) else f"kD={value:g}", config, "kD", value, "W m-1 K-1"))
        return output
    raise ValueError(f"Unsupported study figure: {number}")


def plot_study(number: int, preset: str, output_dir: Path, dpi: int) -> None:
    runs = run_simulation_sweep(make_cases(preset, number), verbose=True)
    plot_parameter_study(
        number,
        runs,
        output_dir,
        dpi,
        reverse_x=number in {16, 18},
        reference_temperature_c=469.0 if number == 15 else 260.0,
    )


def plot_fig19(preset: str, output_dir: Path, dpi: int) -> None:
    base = base_config(preset, n_cells=6, t_end_s=4000.0)
    enhanced_barrier_thickness_m = 0.002
    cases = [
        SimulationCase("400 kJ, 26°C", with_thermal_barrier(replace(base, ambient_temperature_k=299.15), enhanced_barrier_thickness_m, 0.04), "case", 1.0, ""),
        SimulationCase("320 kJ, 17°C", with_thermal_barrier(replace(base, ambient_temperature_k=290.15, trigger=replace(base.trigger, trigger_energy_j=320_000.0)), enhanced_barrier_thickness_m, 0.04), "case", 2.0, ""),
    ]
    runs = run_simulation_sweep(cases, verbose=True)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True, constrained_layout=True)
    arrays: dict[str, np.ndarray] = {}
    for index, (axis, run) in enumerate(zip(axes, runs, strict=True)):
        center = extract_center_temperatures(run.result.Y, run.result.params) - 273.15
        for cell in range(center.shape[1]):
            axis.plot(run.result.t, center[:, cell], color=CELL_COLORS[cell], label=f"T{cell + 1}")
        axis.set(title=run.case.label, xlabel="Time (s)", ylabel="Temperature (°C)")
        axis.set_xlim(0.0, 2000.0)
        axis.legend(ncol=2, fontsize=7)
        style_axis(axis, "ab"[index])
        arrays[f"case_{index + 1}_time_s"] = run.result.t
        arrays[f"case_{index + 1}_center_c"] = center
    finish_figure(figure, 19, output_dir, dpi)
    np.savez_compressed(output_dir / "fig19_data.npz", **arrays)


PLOTTERS: dict[int, Callable[[str, Path, int], None]] = {
    10: plot_fig10,
    11: plot_fig11,
    12: plot_fig12,
    13: plot_fig13,
    14: plot_fig14,
    15: lambda preset, output, dpi: plot_study(15, preset, output, dpi),
    16: lambda preset, output, dpi: plot_study(16, preset, output, dpi),
    17: lambda preset, output, dpi: plot_study(17, preset, output, dpi),
    18: lambda preset, output, dpi: plot_study(18, preset, output, dpi),
    19: plot_fig19,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproduce the simulation content of Feng et al. (2015) Figs. 10-19.")
    parser.add_argument("--figures", type=parse_figures, default=parse_figures("10-19"))
    parser.add_argument("--preset", choices=("fast", "default", "calibration"), default="default")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir or project_root / "results" / "fengxuning"
    output_dir.mkdir(parents=True, exist_ok=True)
    for number in args.figures:
        print(f"Rendering Fig. {number} ({args.preset})")
        PLOTTERS[number](args.preset, output_dir, args.dpi)
    print(f"Outputs saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
