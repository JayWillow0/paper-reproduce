from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np

from .model import ModelParameters


PALETTE = np.array(
    [
        [0.0000, 0.2784, 0.5569],
        [0.8353, 0.3686, 0.0000],
        [0.0000, 0.5294, 0.3373],
        [0.5961, 0.3059, 0.6392],
        [0.7686, 0.4941, 0.0235],
        [0.1373, 0.5333, 0.7333],
        [0.6353, 0.0784, 0.1843],
        [0.3333, 0.3333, 0.3333],
    ]
)
INTERFACE_PALETTE = np.array(
    [[0.1843, 0.4392, 0.6510], [0.8863, 0.4471, 0.1725], [0.2431, 0.5922, 0.4275], [0.4941, 0.3804, 0.6784], [0.7333, 0.6000, 0.2000]]
)


def nature_plot_style(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    style = {
        "font_name": "Arial",
        "font_size": 9,
        "label_font_size": 10,
        "title_font_size": 11,
        "legend_font_size": 8,
        "line_width": 1.7,
        "axis_line_width": 0.8,
        "figure_color": "white",
        "show_grid": False,
        "grid_alpha": 0.10,
        "tr_onset_c": 260.0,
        "show_tr_onset": True,
        "export_path": None,
        "dpi": 150,
        "palette": PALETTE,
        "interface_palette": INTERFACE_PALETTE,
        "threshold_color": (0.7255, 0.1294, 0.1412),
    }
    if overrides:
        unknown = set(overrides) - set(style)
        if unknown:
            raise ValueError(f"Unknown plot style option(s): {', '.join(sorted(unknown))}.")
        style.update(overrides)
    return style


def _style_axis(axis: Any, style: Mapping[str, Any]) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(style["axis_line_width"])
    axis.spines["bottom"].set_linewidth(style["axis_line_width"])
    axis.tick_params(direction="out", labelsize=style["font_size"], width=style["axis_line_width"])
    if style["show_grid"]:
        axis.grid(True, alpha=style["grid_alpha"])
    else:
        axis.grid(False)


def _threshold(axis: Any, style: Mapping[str, Any]) -> None:
    if style["show_tr_onset"]:
        axis.axhline(style["tr_onset_c"], linestyle="--", linewidth=1.0, color=style["threshold_color"])
        axis.text(
            0.01,
            style["tr_onset_c"],
            f"TR onset {style['tr_onset_c']:g}°C",
            transform=axis.get_yaxis_transform(),
            va="bottom",
            fontsize=style["legend_font_size"],
            color=style["threshold_color"],
        )


def _export(figure: Figure, style: Mapping[str, Any]) -> None:
    export_path = style.get("export_path")
    if not export_path:
        return
    path = Path(export_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=style["dpi"], facecolor=style["figure_color"])


def plot_temperatures(
    time_s: np.ndarray,
    center_temperature_k: np.ndarray,
    battery_interface_temperature_k: np.ndarray,
    style_options: Mapping[str, Any] | None = None,
) -> Figure:
    style = nature_plot_style(style_options)
    time_s = np.asarray(time_s, dtype=float)
    center_c = np.asarray(center_temperature_k, dtype=float) - 273.15
    interface_c = np.asarray(battery_interface_temperature_k, dtype=float) - 273.15
    figure, axes = plt.subplots(2, 1, figsize=(1393 / style["dpi"], 946 / style["dpi"]), dpi=style["dpi"], constrained_layout=True)
    figure.patch.set_facecolor(style["figure_color"])
    for cell in range(center_c.shape[1]):
        axes[0].plot(time_s, center_c[:, cell], linewidth=style["line_width"], color=style["palette"][cell % len(style["palette"])], label=f"Bat {cell + 1}")
    _threshold(axes[0], style)
    axes[0].set(title="Center Temperature", ylabel="Center Temperature (°C)", xlim=(time_s.min(), time_s.max()))
    axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=style["legend_font_size"])
    if interface_c.ndim == 2 and interface_c.shape[1]:
        for index in range(interface_c.shape[1]):
            axes[1].plot(time_s, interface_c[:, index], linewidth=style["line_width"], color=style["interface_palette"][index % len(style["interface_palette"])], label=f"Bat {index + 1}-{index + 2}")
        _threshold(axes[1], style)
        axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=style["legend_font_size"])
    else:
        axes[1].text(0.5, 0.5, "No Interface", transform=axes[1].transAxes, ha="center", color="0.35")
    axes[1].set(title="Interface Temperature between Baterries", xlabel="Time (s)", ylabel="Interface Temperature (°C)", xlim=(time_s.min(), time_s.max()))
    for axis in axes:
        _style_axis(axis, style)
        axis.title.set_fontsize(style["title_font_size"])
    _export(figure, style)
    return figure


def plot_all_interface_temperatures(
    time_s: np.ndarray,
    battery_interface_temperature_k: np.ndarray,
    holder_interface_temperature_k: np.ndarray,
    parameters: ModelParameters,
    style_options: Mapping[str, Any] | None = None,
) -> Figure:
    style = nature_plot_style(style_options)
    time_s = np.asarray(time_s, dtype=float)
    battery_c = np.asarray(battery_interface_temperature_k, dtype=float) - 273.15
    holder_c = np.asarray(holder_interface_temperature_k, dtype=float) - 273.15
    figure, axes = plt.subplots(2, 1, figsize=(1392 / style["dpi"], 945 / style["dpi"]), dpi=style["dpi"], constrained_layout=True)
    for index in range(battery_c.shape[1]):
        axes[0].plot(time_s, battery_c[:, index], linewidth=style["line_width"], color=style["interface_palette"][index % len(style["interface_palette"])], label=f"Bat {index + 1}-{index + 2}")
    if battery_c.shape[1]:
        _threshold(axes[0], style)
        axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=style["legend_font_size"])
    else:
        axes[0].text(0.5, 0.5, "No Interface", transform=axes[0].transAxes, ha="center", color="0.35")
    holder_labels = []
    if parameters.topology.has_holder_front:
        holder_labels.append("Holder-L / Bat 1")
    if parameters.topology.has_holder_back:
        holder_labels.append(f"Bat {parameters.n_batteries} / Holder-R")
    for index in range(holder_c.shape[1]):
        axes[1].plot(time_s, holder_c[:, index], linewidth=style["line_width"], color=style["palette"][index % len(style["palette"])], label=holder_labels[index])
    if holder_c.shape[1]:
        _threshold(axes[1], style)
        axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=style["legend_font_size"])
    else:
        axes[1].text(0.5, 0.5, "No Holder Interface", transform=axes[1].transAxes, ha="center", color="0.35")
    axes[0].set(title="Temperature between Batteries", ylabel="Interface Temperature (°C)", xlim=(time_s.min(), time_s.max()))
    axes[1].set(title="Temperature between Battery and Holder", xlabel="Time (s)", ylabel="Interface Temperature (°C)", xlim=(time_s.min(), time_s.max()))
    for axis in axes:
        _style_axis(axis, style)
    _export(figure, style)
    return figure


def plot_concentrations(
    time_s: np.ndarray,
    states: np.ndarray,
    parameters: ModelParameters,
    node_index: int,
    style_options: Mapping[str, Any] | None = None,
) -> Figure:
    style = nature_plot_style(style_options)
    if not 0 <= node_index < parameters.n_nodes:
        raise ValueError(f"node_index must be in [0, {parameters.n_nodes - 1}].")
    columns = min(3, max(1, parameters.n_reactions))
    rows = int(np.ceil(parameters.n_reactions / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False, figsize=(1608 / style["dpi"], 981 / style["dpi"]), dpi=style["dpi"], constrained_layout=True)
    base = node_index * parameters.states_per_node
    for index in range(parameters.n_reactions):
        axis = axes.flat[index]
        axis.plot(time_s, states[:, base + 1 + index], linewidth=style["line_width"], color=style["palette"][index % len(style["palette"])])
        axis.set(title=parameters.concentration_names[index], xlabel="Time (s)", ylabel="Normalized Concentration", xlim=(np.min(time_s), np.max(time_s)))
        _style_axis(axis, style)
    for axis in axes.flat[parameters.n_reactions :]:
        axis.set_visible(False)
    figure.suptitle(f"Node {node_index + 1} concentration evolution", fontsize=style["title_font_size"] + 1)
    _export(figure, style)
    return figure


def plot_thermal_network(parameters: ModelParameters, style_options: Mapping[str, Any] | None = None) -> Figure:
    style = nature_plot_style(style_options)
    figure, axis = plt.subplots(figsize=(12, 3), dpi=100)
    x = 0.5
    if parameters.topology.has_holder_front:
        axis.add_patch(Rectangle((x, 0.2), 0.4, 0.6, facecolor=(0.7, 0.7, 0.7)))
        axis.text(x + 0.05, 0.5, "Holder", fontsize=7)
        x += 0.6
    for cell in range(parameters.n_batteries):
        axis.add_patch(Rectangle((x, 0.2), 0.8, 0.6, facecolor=(0.8, 0.9, 1.0)))
        axis.text(x + 0.25, 0.5, f"Bat{cell + 1}", fontsize=9)
        axis.plot([x + 0.1, x + 0.7], [0.5, 0.5], "k-", linewidth=1)
        x += 1.0
        if cell < parameters.n_batteries - 1:
            axis.plot([x - 0.2, x + 0.3], [0.5, 0.5], "r--", linewidth=1)
            x += 0.2
    if parameters.topology.has_holder_back:
        x += 0.2
        axis.add_patch(Rectangle((x, 0.2), 0.4, 0.6, facecolor=(0.7, 0.7, 0.7)))
        axis.text(x + 0.05, 0.5, "Holder", fontsize=7)
    axis.set(xlim=(0, x + 1.5), ylim=(0, 1.2), title=f"Thermal network ({parameters.n_batteries} cells, holder={parameters.config.topology.holders})")
    axis.axis("off")
    _export(figure, style)
    return figure
