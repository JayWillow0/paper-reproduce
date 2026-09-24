from moduletr import SensitivityParameter, build_config, run_TR_sensitivity
from moduletr.sensitivity import export_sensitivity_csv


base = build_config(n_batteries=2, holders="none", t_end=80, solver_preset="fast")
records = run_TR_sensitivity(
    base,
    [SensitivityParameter("E_short_total", (3e5, 4e5, 5e5), "J")],
    {"mode": "oat", "verbose": True},
)
export_sensitivity_csv(records, "results/sensitivity.csv")

