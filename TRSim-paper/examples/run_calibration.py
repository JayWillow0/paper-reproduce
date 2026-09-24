from moduletr import FitParameter, build_config, run_TR_calibration, run_simulation


base = build_config(n_batteries=1, holders="none", t_end=30, solver_preset="fast")
synthetic = run_simulation(base)
series = synthetic.to_dict("series")["temperature_series"]
target = {"time_s": series["time_s"], "T_cell1_center_K": series["center_k"][0]}
calibration = run_TR_calibration(
    base,
    target,
    [FitParameter("E_short_total", 3.8e5, 2e5, 6e5, scale=1e5, unit="J")],
    {"algorithm": "fminsearch", "max_iter": 10, "rng_seed": 1},
)
print(calibration.to_dict())

