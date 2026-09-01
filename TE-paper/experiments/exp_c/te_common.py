# Shared helpers for experiment C and the multi-seed holdout audit: reuse exp_b's
# data-construction logic, model, and training loop. Does NOT modify repo/ or exp_b/.

import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP_B_DIR = os.path.join(HERE, "..", "exp_b")
sys.path.insert(0, os.path.abspath(EXP_B_DIR))

from experiment_b import (  # noqa: E402
    dataset_generation_function, CNN2DModel, train_model, set_seed, predict,
    TRAIN_DATA_DIR, TEST_DATA_DIR, TRAIN_NAMES, TEST_NAMES,
    DATASET_LENGTH, DSC_NUM, DELTA_SP,
)

CACHE_DIR = os.path.join(HERE, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def load_battery(name, split="train"):
    """Return (X float32 ndarray, y float32 ndarray, Temp_range ndarray), with npz cache."""
    cache = os.path.join(CACHE_DIR, f"{split}_{name}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["X"], z["y"], z["Trange"]
    data_dir = TRAIN_DATA_DIR if split == "train" else TEST_DATA_DIR
    X, y, Trange = dataset_generation_function(data_dir, f"DSC_{name}", f"ARC_{name}",
                                               DATASET_LENGTH, DSC_NUM, DELTA_SP)
    np.savez_compressed(cache, X=X.astype(np.float32), y=np.asarray(y, dtype=np.float32),
                        Trange=np.asarray(Trange, dtype=np.float32))
    return X.astype(np.float32), np.asarray(y, dtype=np.float32), np.asarray(Trange, dtype=np.float32)


def load_all_train():
    """Return per-battery dict and the concatenated pool in TRAIN_NAMES order
    (same concatenation order as exp_b, so split indices are reproducible)."""
    per = {}
    for name in TRAIN_NAMES:
        X, y, Trange = load_battery(name, "train")
        per[name] = (X, y, Trange)
        print(f"  {name}: {len(y)} samples", flush=True)
    X_all = np.concatenate([per[n][0] for n in TRAIN_NAMES], axis=0)
    y_all = np.concatenate([per[n][1] for n in TRAIN_NAMES], axis=0)
    # per-sample metadata: battery index + cutoff temperature
    batt_all = np.concatenate([np.full(len(per[n][1]), i, dtype=np.int64)
                               for i, n in enumerate(TRAIN_NAMES)])
    tcut_all = np.concatenate([per[n][2] for n in TRAIN_NAMES])
    return per, X_all, y_all, batt_all, tcut_all
