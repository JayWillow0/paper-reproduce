"""从校准运行索引生成Part3校准版文章PNG，只读取真实计算结果。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from pemfc_coldstart.io import load_configuration
from pemfc_coldstart.mesh import build_mesh
from pemfc_coldstart.postprocess import COLORS, configure_style
from pemfc_coldstart.state import build_layout, decode_state


ROOT = Path(__file__).resolve().parents[1]
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})
ISO_COLOR = "#8c6bb1"


def save_png(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")


def load_runs() -> dict[str, Path]:
    index = ROOT / "experiments" / "calibration_runs.json"
    mapping = json.loads(index.read_text())
    return {key: ROOT / value for key, value in mapping["runs"].items()}


def load_reference(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return (np.asarray([float(row["time_s"]) for row in rows]),
            np.asarray([float(row["voltage_v"]) for row in rows]),
            np.asarray([float(row["voltage_uncertainty_v"]) for row in rows]))


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontsize=8, fontweight="bold")
    ax.grid(color="#dddddd", lw=0.45, alpha=0.7)


def decoded_series(run: Path, config: Path):
    case, params, _, _ = load_configuration(config)
    mesh = build_mesh(params, case.resolution); layout = build_layout(mesh, case)
    data = np.load(run / "solution.npz")
    states = [decode_state(data["state"][:, i], mesh, layout, case, params) for i in range(data["time_s"].size)]
    return case, params, mesh, data, states


def figure4(runs: dict[str, Path], assets: Path) -> None:
    configure_style()
    iso = np.load(runs["fig4_cal_combined_t95"] / "solution.npz")
    case, _, mesh, data, states = decoded_series(
        runs["fig4_cal_combined_t95"], ROOT / "experiments/exp_a/config_cal_combined_t95.json")
    th = np.load(runs["fig4_cal_combined_t95_coupled"] / "solution.npz")
    ref_t, ref_v, ref_e = load_reference(ROOT / "experiments/exp_a/reference/fig4_experimental_digitized.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7), constrained_layout=True)
    axes[0].plot(data["time_s"], data["voltage_v"], color=COLORS["voltage"], lw=1.5,
                 label="Calibrated isothermal")
    axes[0].plot(th["time_s"], th["voltage_v"], color="#5b9a8b", lw=1.1,
                 ls="--", label="Calibrated thermal")
    axes[0].errorbar(ref_t, ref_v, yerr=ref_e, fmt="o", ms=2.7, mfc="white", mec=COLORS["paper"],
                     mew=0.7, ecolor="#bbbbbb", elinewidth=0.5, label="Digitized experiment")
    axes[0].set(xlabel="Time (s)", ylabel="Cell voltage (V)", xlim=(0, 95), ylim=(0, 0.85))
    axes[0].legend(fontsize=6)
    cl = mesh.cathode_cl_cells
    positions = [0, round(.25*(len(cl)-1)), round(.5*(len(cl)-1)), round(.75*(len(cl)-1)), len(cl)-1]
    labels = ["Membrane", "1/4 CL", "Center", "3/4 CL", "MPL"]
    colors = plt.cm.viridis(np.linspace(0.08, 0.9, 5))
    for index, label, color in zip(positions, labels, colors):
        axes[1].plot(data["time_s"], [s.lambda_n[cl[index]] for s in states], lw=1.15, label=label, color=color)
    axes[1].axhline(7.305862, color="#d95f02", ls=":", lw=1.0, label="Saturation at −20 °C")
    axes[1].set(xlabel="Time (s)", ylabel="Cathode CL water content", xlim=(0, 77))
    axes[1].legend(fontsize=5.5, ncol=2)
    panel_label(axes[0], "a"); panel_label(axes[1], "b")
    save_png(fig, assets / "part3_fig4_calibrated"); plt.close(fig)


def figure5(runs: dict[str, Path], assets: Path) -> None:
    configure_style()
    iso = np.load(runs["fig5_cal_verification"] / "solution.npz")
    th = np.load(runs["fig5_cal_verification_coupled"] / "solution.npz")
    ref_t, ref_v, ref_e = load_reference(ROOT / "experiments/exp_b/reference/fig5_experimental_digitized.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), constrained_layout=True)
    axes[0].plot(iso["time_s"], iso["voltage_v"], color=COLORS["voltage"], lw=1.4,
                 label="Calibrated isothermal")
    axes[0].plot(th["time_s"], th["voltage_v"], color="#5b9a8b", lw=1.1, ls="--",
                 label="Calibrated thermal (prefix)")
    axes[0].errorbar(ref_t, ref_v, yerr=ref_e, fmt="o", ms=2.4, mfc="white", mec=COLORS["paper"], mew=.7,
                     ecolor="#bbbbbb", elinewidth=.5, label="Digitized experiment")
    axes[0].set(xlabel="Time (s)", ylabel="Cell voltage (V)", xlim=(0, 100), ylim=(0, 0.85))
    axes[0].legend(fontsize=5.2, loc="lower left", frameon=True, framealpha=0.9)
    axes[1].plot(iso["time_s"], iso["lambda_cathode_mean"], color=COLORS["lambda"], lw=1.4, label="Isothermal")
    axes[1].plot(th["time_s"], th["lambda_cathode_mean"], color="#5b9a8b", lw=1.0, ls="--", label="Thermal")
    axes[1].axhline(7.305862, color="#d95f02", ls=":", lw=1.0)
    axes[1].set(xlabel="Time (s)", ylabel="Mean CL water content"); axes[1].legend(fontsize=5.5)
    axes[2].plot(iso["time_s"], iso["ice_cathode_mean"], color=COLORS["ice"], lw=1.4, label="Isothermal")
    axes[2].plot(th["time_s"], th["ice_cathode_mean"], color="#5b9a8b", lw=1.0, ls="--", label="Thermal")
    axes[2].set(xlabel="Time (s)", ylabel="Mean CL ice fraction", ylim=(0, 1)); axes[2].legend(fontsize=5.5)
    for label, ax in zip("abc", axes): panel_label(ax, label)
    save_png(fig, assets / "part3_fig5_calibrated"); plt.close(fig)


def figure8(runs: dict[str, Path], assets: Path) -> None:
    configure_style()
    series = [("κ = 1 s⁻¹", runs["fig8_cal_k1"], "#1f4e79"),
              ("κ = 2 s⁻¹", runs["fig8_cal_k2"], "#5b9a8b"),
              ("κ = 3 s⁻¹", runs["fig8_cal_k3"], "#d97706")]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    for label, run, color in series:
        data = np.load(run / "solution.npz")
        axes[0].plot(data["time_s"], data["voltage_v"], color=color, lw=1.2, label=label)
        axes[1].plot(data["time_s"], data["lambda_cathode_max"], color=color, lw=1.2)
        axes[2].plot(data["time_s"], data["ice_cathode_max"], color=color, lw=1.2)
    axes[0].set(xlabel="Time (s)", ylabel="Cell voltage (V)", ylim=(0.3, 0.7)); axes[0].legend(fontsize=5.1)
    axes[1].set(xlabel="Time (s)", ylabel="Maximum CL water content")
    axes[2].set(xlabel="Time (s)", ylabel="Maximum CL ice fraction", ylim=(0, 1))
    for label, ax in zip("abc", axes): panel_label(ax, label)
    save_png(fig, assets / "part3_fig8_calibrated"); plt.close(fig)


def figure9(runs: dict[str, Path], assets: Path) -> None:
    configure_style()
    specs = [
        ("Dissolved", runs["fig9_cal_dissolved"], ROOT / "experiments/exp_d/config_cal_dissolved.json",
         [9, 25, 44, 58], "#1f4e79"),
        ("Vapor", runs["fig9_cal_vapor"], ROOT / "experiments/exp_d/config_cal_vapor.json",
         [9, 25, 45, 70], "#d97706"),
        ("Liquid", runs["fig9_cal_liquid"], ROOT / "experiments/exp_d/config_cal_liquid.json",
         [9, 20, 30, 36], "#5b9a8b"),
    ]
    fig = plt.figure(figsize=(7.1, 5.3), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.05, 1.05])
    axv = fig.add_subplot(grid[0, :])
    for label, run, config, _, color in specs:
        data = np.load(run / "solution.npz")
        axv.plot(data["time_s"], data["voltage_v"], color=color, lw=1.4, label=label)
    shutdown = {"Dissolved": 62.0, "Vapor": 48.0, "Liquid": 40.0}
    for (label, _, _, _, color), t_stop in zip(specs, [shutdown[s[0]] for s in specs]):
        axv.axvline(t_stop, color=color, ls=":", lw=0.8)
    axv.set(xlabel="Time (s)", ylabel="Cell voltage (V)", xlim=(0, 80), ylim=(0.3, 0.7))
    axv.legend(ncol=3, fontsize=7, loc="lower left")
    panel_label(axv, "a")
    for col, (label, run, config, snapshots, color) in enumerate(specs):
        case, params, mesh, data, states = decoded_series(run, config)
        cl = mesh.cathode_cl_cells; xi = (mesh.x_m[cl] - mesh.face_x_m[cl[0]]) * 1e6
        ax_l = fig.add_subplot(grid[1, col]); ax_i = fig.add_subplot(grid[2, col])
        shades = plt.cm.Blues(np.linspace(.35, .9, len(snapshots))) if label == "Dissolved" else (
                 plt.cm.Oranges(np.linspace(.35, .9, len(snapshots))) if label == "Vapor" else plt.cm.Greens(np.linspace(.35, .9, len(snapshots))))
        for time, shade in zip(snapshots, shades):
            index = int(np.argmin(np.abs(data["time_s"] - time)))
            ax_l.plot(xi, states[index].saturation_liquid[cl], color=shade, lw=1.1, label=f"{data['time_s'][index]:g} s")
            ax_i.plot(xi, states[index].saturation_ice[cl], color=shade, lw=1.1)
        ax_l.set_title(label, fontsize=7); ax_l.set(xlabel="CL location (µm)",
                                                   ylabel="Liquid fraction" if col == 0 else "",
                                                   ylim=(0, 0.03))
        ax_i.set(xlabel="CL location (µm)", ylabel="Ice fraction" if col == 0 else "", ylim=(0, 1))
        ax_l.legend(fontsize=5.2); ax_l.grid(color="#dddddd", lw=.4); ax_i.grid(color="#dddddd", lw=.4)
        panel_label(ax_l, chr(ord("b") + col)); panel_label(ax_i, chr(ord("e") + col))
    save_png(fig, assets / "part3_fig9_calibrated"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args(); args.assets.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    figure4(runs, args.assets); figure5(runs, args.assets)
    figure8(runs, args.assets); figure9(runs, args.assets)


if __name__ == "__main__": main()
