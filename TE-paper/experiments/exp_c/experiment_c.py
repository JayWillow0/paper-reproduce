# Experiment C: isolate slice leakage on the 11 Train batteries.
#   C0  : sample-level random 80/20 baseline, trained in this run.
#   C1a : random 20% val, then drop train slices within <1 degC (same battery) of any val slice
#         -> keeps only 1.9% of train samples (structural: val cutoff spacing ~0.5 degC)
#   C1b : buffer <5 degC -> keeps 0 samples, INFEASIBLE as specified (documented, not trained)
#   C1-0.25C / C1-0.5C : added intermediate buffers so the buffer-width gradient is measurable
#   ctrl: random-subset controls matched to each pruned train-set size (disentangles
#         data starvation from leak removal)
#   C2  : exploratory leave-one-battery-out on NCM622 (train on the other 10).
#         This is a single deliberately difficult group and cannot identify a
#         battery-recognition mechanism by itself.
# Same model, seed=42, 50 epochs throughout. Reuses exp_b code via te_common.
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

from te_common import (load_all_train, train_model, set_seed, predict,
                       TRAIN_NAMES, CNN2DModel)  # noqa: F401

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-4
LOO_NAME = "NCM622"


def r_split_indices(n, seed=SEED):
    """Reproduce exp_b split R exactly: random_split([0.8n, rest], generator=manual_seed(seed))
    is internally randperm(n, generator); first 80% = train, rest = val."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_train = int(0.8 * n)
    return perm[:n_train], perm[n_train:]


def buffer_prune(train_idx, val_idx, batt_all, tcut_all, buffer):
    """Drop train samples sharing battery with and within `buffer` degC of any val sample."""
    keep = np.ones(len(train_idx), dtype=bool)
    for b in np.unique(batt_all[val_idx]):
        v_t = np.sort(tcut_all[val_idx[batt_all[val_idx] == b]])
        tr_mask = batt_all[train_idx] == b
        tr_t = tcut_all[train_idx[tr_mask]]
        pos = np.searchsorted(v_t, tr_t)
        d = np.full(len(tr_t), np.inf)
        for k, p in ((0, pos - 1), (1, pos)):
            valid = (p >= 0) & (p < len(v_t))
            d[valid] = np.minimum(d[valid], np.abs(tr_t[valid] - v_t[p[valid]]))
        drop_local = d < buffer
        keep[np.where(tr_mask)[0][drop_local]] = False
    return train_idx[keep]


def eval_mse_on_indices(model, X_all, y_all, idx):
    X_t = torch.tensor(X_all[idx], dtype=torch.float32)
    pred = predict(model, X_t)
    return float(np.mean((pred - y_all[idx]) ** 2))


def run(tag, train_idx, val_idx, X_all, y_all):
    t0 = time.time()
    tr_ds = TensorDataset(torch.tensor(X_all[train_idx]), torch.tensor(y_all[train_idx]))
    va_ds = TensorDataset(torch.tensor(X_all[val_idx]), torch.tensor(y_all[val_idx]))
    model, tr_loss, va_loss = train_model(tr_ds, va_ds, EPOCHS, LR, BATCH_SIZE, SEED, tag=tag)
    mse = eval_mse_on_indices(model, X_all, y_all, val_idx)
    secs = round(time.time() - t0, 1)
    print(f"[{tag}] final-epoch val MSE = {mse:.4f} ({secs}s)", flush=True)
    return model, mse, tr_loss, va_loss, secs


def main():
    t_start = time.time()
    print("Loading train-battery datasets ...", flush=True)
    per, X_all, y_all, batt_all, tcut_all = load_all_train()
    n = len(y_all)
    print(f"Total: {n}", flush=True)

    train_idx, val_idx = r_split_indices(n)
    assert len(val_idx) == n - int(0.8 * n)
    print(f"Split R reproduced: train {len(train_idx)}, val {len(val_idx)}", flush=True)

    log = {"protocol_version": 2,
           "protocol": "all conditions trained in one run with the same fixed 80/20 split and sample-weighted MSE",
           "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
           "n_total": n, "n_val": int(len(val_idx)),
           "loo_battery": LOO_NAME, "results": {}}

    # ---- C0 baseline: train in this run so no legacy metric is mixed in ----
    model0, mse0, tr_l0, va_l0, secs0 = run("C0", train_idx, val_idx, X_all, y_all)
    log["results"]["C0"] = {"n_train": int(len(train_idx)), "val_mse": mse0,
                              "seconds": secs0, "train_loss_final": tr_l0[-1],
                              "val_loss_curve": va_l0}
    torch.save(model0.state_dict(), os.path.join(HERE, "model_C0.pt"))

    # ---- C1 buffer ladder (+ size-matched random-subset controls) ----
    # NOTE on the spec'd buffers: with delta_SP=0.1 and a 20% random val set, mean
    # spacing between val cutoff temps is ~0.5 degC, so buffer<1C keeps only 1.9%
    # of train samples and buffer<5C keeps 0 (structural, verified before training).
    # C1b(5C) is therefore infeasible as specified; we add intermediate buffers
    # 0.25C / 0.5C so the "MSE vs buffer width" gradient is actually measurable.
    for tag, buffer in (("C1-0.25C", 0.25), ("C1-0.5C", 0.5), ("C1a", 1.0)):
        pruned = buffer_prune(train_idx, val_idx, batt_all, tcut_all, buffer)
        frac = len(pruned) / len(train_idx)
        print(f"{tag}: buffer<{buffer}C keeps {len(pruned)}/{len(train_idx)} "
              f"train samples ({frac:.1%})", flush=True)
        model, mse, tr_l, va_l, secs = run(tag, pruned, val_idx, X_all, y_all)
        log["results"][tag] = {"buffer_degc": buffer, "n_train_kept": int(len(pruned)),
                               "kept_frac_of_train80": round(frac, 4),
                               "kept_frac_of_total": round(len(pruned) / n, 4),
                               "val_mse": mse, "seconds": secs,
                               "train_loss_final": tr_l[-1], "val_loss_curve": va_l}
        torch.save(model.state_dict(), os.path.join(HERE, f"model_{tag}.pt"))

        # size-matched control: random subset of the original 80% train pool
        rng = np.random.RandomState(SEED)
        ctrl_idx = rng.choice(train_idx, size=len(pruned), replace=False)
        model_c, mse_c, _, _, secs_c = run(f"{tag}-ctrl", ctrl_idx, val_idx, X_all, y_all)
        log["results"][f"{tag}-ctrl"] = {"n_train": int(len(ctrl_idx)),
                                         "val_mse": mse_c, "seconds": secs_c}

    # C1b as specified (buffer<5C): 0 training samples survive -> infeasible
    pruned_5 = buffer_prune(train_idx, val_idx, batt_all, tcut_all, 5.0)
    log["results"]["C1b"] = {"buffer_degc": 5.0, "n_train_kept": int(len(pruned_5)),
                             "infeasible": True,
                             "note": "buffer<5C removes 100% of train samples; cannot train"}

    # ---- C2: leave-one-battery-out (NCM622) ----
    loo_batt_idx = TRAIN_NAMES.index(LOO_NAME)
    tr_mask = batt_all != loo_batt_idx
    tr_all_idx = np.where(tr_mask)[0]
    # Internal 20% validation from the 10-battery pool, matching exp_b v2.
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(tr_all_idx), generator=g).numpy()
    train_n = int(0.8 * len(tr_all_idx))
    c2_train, c2_mon = tr_all_idx[perm[:train_n]], tr_all_idx[perm[train_n:]]
    t0 = time.time()
    tr_ds = TensorDataset(torch.tensor(X_all[c2_train]), torch.tensor(y_all[c2_train]))
    mon_ds = TensorDataset(torch.tensor(X_all[c2_mon]), torch.tensor(y_all[c2_mon]))
    model2, tr_l2, mon_l2 = train_model(tr_ds, mon_ds, EPOCHS, LR, BATCH_SIZE, SEED, tag="C2")
    loo_idx = np.where(~tr_mask)[0]
    mse2 = eval_mse_on_indices(model2, X_all, y_all, loo_idx)
    secs2 = round(time.time() - t0, 1)
    print(f"[C2] LOO({LOO_NAME}) MSE = {mse2:.4f} ({secs2}s)", flush=True)
    log["results"]["C2"] = {"loo_battery": LOO_NAME, "n_train": int(len(c2_train)),
                            "n_loo_eval": int(len(loo_idx)), "loo_mse": mse2,
                            "monitor_mse_final": mon_l2[-1], "seconds": secs2}
    torch.save(model2.state_dict(), os.path.join(HERE, "model_C2.pt"))

    # NCM622 pred vs truth curve
    X_loo, y_loo, Trange_loo = per[LOO_NAME]
    pred_loo = predict(model2, torch.tensor(X_loo, dtype=torch.float32))
    plt.figure(figsize=(10, 4))
    plt.plot(Trange_loo, y_loo, label="Ground truth log(dT/dt)", lw=1.5)
    plt.plot(Trange_loo, pred_loo, label="Predicted (C2 LOO)", lw=1.2, alpha=0.8)
    plt.xlabel("Cutoff temperature (°C)")
    plt.ylabel("log(dT/dt)")
    plt.title(f"C2 leave-one-battery-out: {LOO_NAME} — MSE {mse2:.3f}")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "exp_c_loo_NCM622_curve.png"), dpi=150)
    plt.close()

    # ---- summary bar chart ----
    order = ["C0", "C1-0.25C", "C1-0.25C-ctrl", "C1-0.5C", "C1-0.5C-ctrl", "C1a", "C1a-ctrl", "C2"]
    vals = [mse0,
            log["results"]["C1-0.25C"]["val_mse"], log["results"]["C1-0.25C-ctrl"]["val_mse"],
            log["results"]["C1-0.5C"]["val_mse"], log["results"]["C1-0.5C-ctrl"]["val_mse"],
            log["results"]["C1a"]["val_mse"], log["results"]["C1a-ctrl"]["val_mse"],
            mse2]
    labels = ["C0\nrandom 80/20", "C1 buf<0.25°C", "ctrl\n(size-matched)",
              "C1 buf<0.5°C", "ctrl\n(size-matched)", "C1a buf<1°C",
              "ctrl\n(size-matched)", f"C2\nLOO {LOO_NAME}"]
    colors = ["#4C9BD6", "#F2B880", "#9E9E9E", "#E07B54", "#9E9E9E", "#C0504D", "#9E9E9E", "#8064A2"]
    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("MSE (log(dT/dt))")
    plt.title(f"Experiment C: leak-isolation splits on 11 train batteries (seed={SEED}, {EPOCHS} epochs)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "exp_c_mse_comparison.png"), dpi=150)
    plt.close()

    log["total_seconds"] = round(time.time() - t_start, 1)
    with open(os.path.join(HERE, "experiment_c_metrics.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n==== SUMMARY ====")
    print(f"C0 baseline: {mse0:.4f}")
    for tag in ("C1-0.25C", "C1-0.25C-ctrl", "C1-0.5C", "C1-0.5C-ctrl", "C1a", "C1a-ctrl"):
        r = log["results"][tag]
        nn_ = r.get("n_train_kept", r.get("n_train"))
        print(f"{tag}: val MSE {r['val_mse']:.4f} (train n={nn_})")
    print(f"C1b (buffer<5C): INFEASIBLE, 0 train samples kept")
    print(f"C2 LOO({LOO_NAME}): MSE {mse2:.4f}")
    print(f"Total runtime: {log['total_seconds']} s", flush=True)


if __name__ == "__main__":
    main()
