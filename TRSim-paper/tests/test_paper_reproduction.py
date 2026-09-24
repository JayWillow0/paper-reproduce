from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from moduletr import SimulationCase, run_simulation_sweep


def load_reproduction_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "fengxuning_paper_reproduce.py"
    spec = importlib.util.spec_from_file_location("fengxuning_paper_reproduce", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_figure_selection_and_study_cases_cover_paper_scope() -> None:
    module = load_reproduction_module()
    assert module.parse_figures("10,12-14,19") == [10, 12, 13, 14, 19]
    assert set(module.PLOTTERS) == set(range(10, 20))
    assert len(module.make_cases("fast", 15)) == 6
    assert len(module.make_cases("fast", 17)) == 7


def test_fig11_fast_export_structure(tmp_path: Path) -> None:
    module = load_reproduction_module()
    module.plot_fig11("fast", tmp_path, 60)
    assert (tmp_path / "fig11.png").is_file()
    assert (tmp_path / "fig11.svg").is_file()
    assert (tmp_path / "fig11_data.npz").is_file()


def test_parameter_study_uses_nested_panel_a_and_requested_time_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_reproduction_module()
    cases = [
        SimulationCase(
            f"case {index}",
            module.base_config("fast", n_cells=6, t_end_s=1.0),
            "parameter",
            float(index),
            "-",
        )
        for index in range(4)
    ]
    runs = run_simulation_sweep(cases)
    captured = {}

    def capture(figure, number, output_dir, dpi):
        captured["figure"] = figure

    monkeypatch.setattr(module, "finish_figure", capture)
    module.plot_parameter_study(16, runs, tmp_path, 60)
    figure = captured["figure"]
    assert len(figure.axes) == 7
    for axis in figure.axes[:4]:
        assert axis.get_xlim() == pytest.approx((0.0, 500.0))
    assert figure.axes[5].get_xlim() == pytest.approx((0.0, 500.0))
    assert figure._suptitle is None
    plt.close(figure)
