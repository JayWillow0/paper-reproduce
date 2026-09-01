# Multi-seed reinforcement of the corrected experiment B protocol.
# For each seed, fit one sample-level random 80/20 model on the 11 Train
# batteries, then evaluate the same fitted weights on the three Test batteries.
# Runs from wc-7/work/exp_c/. Does NOT modify repo/ or exp_b/.

import os
import json
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from te_common import (load_all_train, load_battery, train_model, predict,
                       TEST_NAMES, TRAIN_NAMES)  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

SEEDS = [1, 7, 42]
PAPER_TABLE_S8 = {"NCM811_80": 0.47, "NCM811_VC": 0.46, "NCM523": 0.18}
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-4


def main():
    t_start = time.time()
    print("Loading train-battery datasets ...", flush=True)
    per, X_all, y_all, batt_all, tcut_all = load_all_train()
    n = len(y_all)

    log = {"protocol_version": 2,
           "protocol": "one fixed 80/20 sample-level model per seed, evaluated on battery holdouts",
           "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "runs": {}}

    for seed in SEEDS:
        # Match exp_b exactly: the first 80% of randperm is training data and
        # the remaining 20% is validation data.
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).numpy()
        train_n = int(0.8 * n)
        tr_idx, mon_idx = perm[:train_n], perm[train_n:]
        tr_ds = TensorDataset(torch.tensor(X_all[tr_idx]), torch.tensor(y_all[tr_idx]))
        mon_ds = TensorDataset(torch.tensor(X_all[mon_idx]), torch.tensor(y_all[mon_idx]))

        t0 = time.time()
        model, tr_l, mon_l = train_model(tr_ds, mon_ds, EPOCHS, LR, BATCH_SIZE, seed, tag=f"shared-s{seed}")
        res = {"monitor_mse_final": mon_l[-1], "seconds": round(time.time() - t0, 1)}
        for name in TEST_NAMES:
            X, y, Trange = load_battery(name, "test")
            pred = predict(model, torch.tensor(X, dtype=torch.float32))
            mse = float(np.mean((pred - y) ** 2))
            res[name] = mse
            print(f"  seed={seed} {name}: MSE = {mse:.4f}", flush=True)
        log["runs"][str(seed)] = res
        torch.save(model.state_dict(), os.path.join(HERE, f"model_shared_seed{seed}.pt"))

    # ---- summary with all 3 seeds ----
    all_seeds = {str(s): log["runs"][str(s)] for s in SEEDS}
    summary = {}
    for name in TEST_NAMES:
        seed_keys = [str(s) for s in SEEDS]
        vals = [all_seeds[s][name] for s in seed_keys]
        summary[name] = {"mse_per_seed": {s: all_seeds[s][name] for s in seed_keys},
                         "mean": float(np.mean(vals)), "min": float(np.min(vals)),
                         "max": float(np.max(vals)), "paper_table_S8": PAPER_TABLE_S8[name]}
    log["summary_3seeds"] = summary
    log["total_seconds"] = round(time.time() - t_start, 1)

    with open(os.path.join(HERE, "exp_b_multiseed_metrics.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    # grouped bar chart with min-max error bars
    x = np.arange(len(TEST_NAMES))
    means = [summary[n]["mean"] for n in TEST_NAMES]
    yerr_lo = [summary[n]["mean"] - summary[n]["min"] for n in TEST_NAMES]
    yerr_hi = [summary[n]["max"] - summary[n]["mean"] for n in TEST_NAMES]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(x, means, yerr=[yerr_lo, yerr_hi], capsize=6, color="#E07B54",
                   error_kw=dict(ecolor="k", lw=1.2))
    for i, nme in enumerate(TEST_NAMES):
        for j, s in enumerate([str(seed) for seed in SEEDS]):
            v = summary[nme]["mse_per_seed"][s]
            plt.scatter(x[i] + (j - 1) * 0.12, v, color="k", s=18, zorder=3)
        plt.hlines(PAPER_TABLE_S8[nme], x[i] - 0.4, x[i] + 0.4, colors="b",
                   linestyles="dotted", label="paper Table S8" if i == 0 else None)
        plt.text(x[i], means[i] + yerr_hi[i] + 0.05, f"{means[i]:.3f}", ha="center", fontsize=9)
    plt.xticks(x, TEST_NAMES)
    plt.ylabel("MSE (log(dT/dt))")
    plt.title("Shared 80/20 model evaluated on battery holdouts, seeds 1/7/42")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "exp_b_multiseed_shared.png"), dpi=150)
    plt.close()

    print("\n==== SUMMARY (seeds 1/7/42) ====")
    for nme in TEST_NAMES:
        s = summary[nme]
        print(f"{nme}: mean {s['mean']:.4f}, range [{s['min']:.4f}, {s['max']:.4f}], paper {s['paper_table_S8']}")
    print(f"Total runtime: {log['total_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
