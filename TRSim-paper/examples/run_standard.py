from __future__ import annotations

from pathlib import Path

from moduletr import (
    build_config,
    calculate_all_interface_temperatures,
    extract_center_temperatures,
    plot_all_interface_temperatures,
    plot_concentrations,
    plot_temperatures,
    run_TR_model,
)


config = build_config(
    n_batteries=5,
    holders="both",
    active_reactions=["SEI", "anode", "separator", "electrolyte", "cathode1", "cathode2"],
    trigger_type=1,
    trigger_battery=1,
    heating_stop_mode=1,
    t_end=2000,
    solver_preset="default",
)
result = run_TR_model(config, {"verbose": True})
center = extract_center_temperatures(result.Y, result.params)
battery_interface, holder_interface = calculate_all_interface_temperatures(result.Y, result.params)
results = Path(__file__).resolve().parents[1] / "results"
plot_temperatures(result.t, center, battery_interface, {"export_path": results / "temp.png"})
plot_all_interface_temperatures(result.t, battery_interface, holder_interface, result.params, {"export_path": results / "temp_inter.png"})
plot_concentrations(result.t, result.Y, result.params, result.params.node_map["bat5_b"], {"export_path": results / "conc.png"})

