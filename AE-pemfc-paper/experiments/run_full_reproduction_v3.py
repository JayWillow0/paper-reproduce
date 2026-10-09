"""按式(11)局部C2连续化协议统一运行图4、5、8、9。"""

from __future__ import annotations

import json
from pathlib import Path
import time

from pemfc_coldstart.diagnostics import validate_run
from pemfc_coldstart.io import (create_run_dir, load_configuration,
                                refresh_manifest, save_result)
from pemfc_coldstart.parameters import MODEL_PROTOCOL
from pemfc_coldstart.postprocess import plot_run
from pemfc_coldstart.solver import simulate


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "experiments/full_reproduction_v3_runs.json"
CASES = {
    "exp_a_isothermal": "experiments/exp_a/config.json",
    "exp_a_coupled": "experiments/exp_a/config_coupled.json",
    "exp_b_isothermal": "experiments/exp_b/config.json",
    "exp_b_coupled": "experiments/exp_b/config_coupled.json",
    "exp_c_kappa1": "experiments/exp_c/config.json",
    "exp_c_kappa2": "experiments/exp_c/config_kappa2.json",
    "exp_c_kappa3": "experiments/exp_c/config_kappa3.json",
    "exp_c_equilibrium_coarse": "experiments/exp_c/config_equilibrium.json",
    "exp_d_dissolved": "experiments/exp_d/config.json",
    "exp_d_vapor": "experiments/exp_d/config_vapor.json",
    "exp_d_liquid": "experiments/exp_d/config_liquid.json",
}


def write_index(records: dict[str, object]) -> None:
    payload = {"schema": 1, "model_protocol": MODEL_PROTOCOL, "runs": records}
    INDEX.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    records: dict[str, object] = {}
    total_started = time.monotonic()
    for number, (key, relative_config) in enumerate(CASES.items(), 1):
        config = ROOT / relative_config
        case, params, options, raw = load_configuration(config)
        print(f"[{number}/{len(CASES)}] START {key} {relative_config}", flush=True)
        started = time.monotonic()
        run_dir = create_run_dir(config.parent, raw)
        result = simulate(case, params, options)
        save_result(run_dir, result, case, params, options, raw, config)
        plot_run(run_dir)
        refresh_manifest(run_dir)
        validation = validate_run(run_dir)
        if not validation["valid"]:
            raise RuntimeError(f"{key}输出完整性检查失败 {validation}")
        records[key] = {
            "run": str(run_dir.relative_to(ROOT)),
            "status": result.status,
            "end_time_s": float(result.time_s[-1]),
            "elapsed_s": time.monotonic() - started,
        }
        write_index(records)
        print(
            f"[{number}/{len(CASES)}] END {key} status={result.status} "
            f"t={result.time_s[-1]:.9g}s elapsed={records[key]['elapsed_s']:.2f}s",
            flush=True,
        )
    print(f"INDEX {INDEX}", flush=True)
    print(f"TOTAL_ELAPSED {time.monotonic() - total_started:.2f}s", flush=True)


if __name__ == "__main__":
    main()
