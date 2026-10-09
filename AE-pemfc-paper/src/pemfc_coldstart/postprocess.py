"""运行结果绘图。所有图由Python/matplotlib生成。"""

from __future__ import annotations

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .parameters import CaseConfig, ModelParameters
from .state import build_layout, decode_state
from .mesh import build_mesh


COLORS = {"voltage": "#1f4e79", "lambda": "#d97706", "ice": "#4c78a8",
          "liquid": "#5b9a8b", "paper": "#777777", "temperature": "#b94a48"}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
        "legend.frameon": False, "xtick.direction": "out", "ytick.direction": "out",
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_run(run_dir: Path) -> list[Path]:
    configure_style()
    data = np.load(run_dir / "solution.npz")
    t = data["time_s"]
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.6), constrained_layout=True)
    axes[0, 0].plot(t, data["voltage_v"], color=COLORS["voltage"], lw=1.5)
    experiment = run_dir.parent.parent
    references = sorted((experiment / "reference").glob("*experimental_digitized.csv"))
    if references:
        with references[0].open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        ref_t = np.asarray([float(row["time_s"]) for row in rows])
        ref_v = np.asarray([float(row["voltage_v"]) for row in rows])
        ref_e = np.asarray([float(row["voltage_uncertainty_v"]) for row in rows])
        axes[0, 0].errorbar(ref_t, ref_v, yerr=ref_e, fmt="o", ms=2.4, mfc="white",
                            mec=COLORS["paper"], mew=0.7, ecolor="#bbbbbb", elinewidth=0.5,
                            label="Digitized experiment")
        axes[0, 0].legend(fontsize=6)
    axes[0, 0].set(xlabel="Time (s)", ylabel="Cell voltage (V)")
    axes[0, 1].plot(t, data["lambda_cathode_mean"], color=COLORS["lambda"], lw=1.5, label="Mean")
    axes[0, 1].plot(t, data["lambda_cathode_max"], color=COLORS["lambda"], lw=1.0, ls="--", label="Maximum")
    axes[0, 1].set(xlabel="Time (s)", ylabel="Cathode CL water content"); axes[0, 1].legend()
    axes[1, 0].plot(t, data["ice_cathode_mean"], color=COLORS["ice"], lw=1.5, label="Ice")
    axes[1, 0].plot(t, data["liquid_cathode_mean"], color=COLORS["liquid"], lw=1.5, label="Liquid")
    axes[1, 0].set(xlabel="Time (s)", ylabel="Mean volume fraction"); axes[1, 0].legend()
    axes[1, 1].plot(t, data["temperature_max_k"] - 273.15, color=COLORS["temperature"], lw=1.5)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Maximum temperature (°C)")
    for label, ax in zip("abcd", axes.flat):
        ax.text(-0.16, 1.05, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
        ax.grid(color="#dddddd", linewidth=0.45, alpha=0.7)
    stem = run_dir / "figures" / "overview"
    save_figure(fig, stem); plt.close(fig)
    return [stem.with_suffix(".png")]


def plot_comparison(run_dirs: list[Path], labels: list[str], output_stem: Path,
                    title: str | None = None) -> list[Path]:
    configure_style()
    palette = ["#1f4e79", "#d97706", "#5b9a8b", "#8c6bb1", "#b94a48", "#6c757d"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.35), constrained_layout=True)
    for i, (run, label) in enumerate(zip(run_dirs, labels)):
        data = np.load(run / "solution.npz")
        color = palette[i % len(palette)]
        axes[0].plot(data["time_s"], data["voltage_v"], lw=1.35, color=color, label=label)
        axes[1].plot(data["time_s"], data["lambda_cathode_max"], lw=1.35, color=color, label=label)
        axes[2].plot(data["time_s"], data["ice_cathode_max"], lw=1.35, color=color, label=label)
    axes[0].set(xlabel="Time (s)", ylabel="Cell voltage (V)")
    axes[1].set(xlabel="Time (s)", ylabel="Maximum CL water content")
    axes[2].set(xlabel="Time (s)", ylabel="Maximum CL ice fraction")
    axes[2].legend(loc="best", fontsize=6)
    for label, ax in zip("abc", axes):
        ax.text(-0.18, 1.04, label, transform=ax.transAxes, fontweight="bold", fontsize=8)
        ax.grid(color="#dddddd", linewidth=0.45, alpha=0.7)
    if title: fig.suptitle(title, fontsize=8)
    save_figure(fig, output_stem); plt.close(fig)
    return [output_stem.with_suffix(".png")]
