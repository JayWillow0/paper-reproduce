# Experiment B: one fixed sample-level 80/20 model, evaluated both on its
# random validation slices (R) and on three battery-level holdouts (H).
# Reuses the data-construction logic and CNN2DModel architecture from
# repo/TE-method-2024/1-model training/Code/supply_function.py (verbatim logic, parameterized paths).
# Runs from wc-7/work/. Does NOT modify repo/.

import os
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(HERE, "..", "..", "repo", "TE-method-2024", "1-model training", "Data")
TEST_DATA_DIR = os.path.join(HERE, "..", "..", "repo", "TE-method-2024", "2-model test", "Data")
FIG_DIR = os.path.join(HERE, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
EPOCHS = int(os.environ.get("EXP_B_EPOCHS", "50"))
BATCH_SIZE = 128
LR = 1e-4
DELTA_SP = 0.1
DATASET_LENGTH = 100
DSC_NUM = 5

TRAIN_NAMES = ["NCA", "NCM83116", "NCM811_100", "NCM811_60", "NCM811_40",
               "NCM811_20", "NCM811_0", "NCM811_FEC", "NCM811_PS", "NCM811_HC", "NCM622"]
TEST_NAMES = ["NCM811_80", "NCM811_VC", "NCM523"]

# Paper Table S8 (held-out battery MSE) for reference
PAPER_TABLE_S8 = {"NCM811_80": 0.47, "NCM811_VC": 0.46, "NCM523": 0.18}


def dataset_generation_function(location, DSC_name, ARC_name, Dataset_length, DSC_num, delta_SP):
    """Same logic as repo supply_function.dataset_generation_function, with
    hardcoded '../data/' and delta_SP=0.1 turned into parameters."""
    filename_DSC = f"{DSC_name}_{DSC_num}.txt"
    DSC_data = np.loadtxt(os.path.join(location, filename_DSC), skiprows=1, delimiter=',')
    Temperature_DSC = DSC_data[:, 0]

    filename_ARC = f"{ARC_name}.txt"
    data = np.loadtxt(os.path.join(location, filename_ARC), skiprows=1, delimiter=',')
    Temperature_ARC = data[:, 1]
    dTdt_ARC = data[:, 2]

    Temp_ARC_l = np.ceil(np.min(Temperature_ARC))
    Temp_ARC_h = np.floor(np.max(Temperature_ARC))
    Temp_DSC_l = np.ceil(np.min(Temperature_DSC))
    Temp_DSC_h = np.floor(np.max(Temperature_DSC))

    Temp_l = 90 if Temp_ARC_l < 90 else Temp_ARC_l
    Temp_h = min(Temp_ARC_h, Temp_DSC_h)

    Temp_range = np.arange(Temp_l, Temp_h + 0.0005, delta_SP).round(3)

    DSC_dataset = np.zeros((len(Temp_range), Dataset_length, DSC_num + 1))
    for i, temp in enumerate(Temp_range):
        Temp_i = np.linspace(Temp_DSC_l, temp, Dataset_length)
        interpolator = interp1d(Temperature_DSC, DSC_data[:, 1:], axis=0, fill_value="extrapolate")
        interpolated_data = interpolator(Temp_i)
        DSC_dataset[i, :, :1] = Temp_i[:, np.newaxis]
        DSC_dataset[i, :, 1:] = interpolated_data

    interpolatorARC = interp1d(Temperature_ARC, dTdt_ARC, fill_value="extrapolate")
    ARC_dTdt = interpolatorARC(Temp_range)
    log_ARC_dTdt = np.log(ARC_dTdt)

    DSC_dataset_norm = np.copy(DSC_dataset)
    DSC_dataset_norm[:, :, 0] = (DSC_dataset_norm[:, :, 0] - 50) / 1000

    return DSC_dataset_norm, log_ARC_dTdt, Temp_range


class CNN2DModel(nn.Module):
    """Verbatim architecture from repo supply_function.model_define_train."""
    def __init__(self):
        super(CNN2DModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=200, out_features=100)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(torch.relu(self.fc1(x)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(train_dataset, val_dataset, epochs, lr, batch_size, seed, tag):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNN2DModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_loss_list, val_loss_list = [], []

    for epoch in range(epochs):
        model.train()
        train_squared_error = 0.0
        train_sample_count = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(1).float())
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_squared_error += loss.item() * labels.numel()
            train_sample_count += labels.numel()
        train_loss_epoch = train_squared_error / train_sample_count
        train_loss_list.append(train_loss_epoch)

        model.eval()
        val_squared_error = 0.0
        val_sample_count = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.unsqueeze(1).float())
                loss = criterion(outputs, labels.float().unsqueeze(1))
                val_squared_error += loss.item() * labels.numel()
                val_sample_count += labels.numel()
        val_loss_epoch = val_squared_error / val_sample_count
        val_loss_list.append(val_loss_epoch)
        print(f"[{tag}] Epoch {epoch + 1}/{epochs} - train_loss: {train_loss_epoch:.5f} - val_loss: {val_loss_epoch:.5f}", flush=True)

    return model, train_loss_list, val_loss_list


def predict(model, data, batch_size=512):
    model.eval()
    loader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (inputs,) in loader:
            preds.append(model(inputs.unsqueeze(1).float()).squeeze(1).numpy())
    return np.concatenate(preds)


def main():
    t_start = time.time()
    log = {"protocol_version": 2,
           "protocol": "one fixed 80/20 sample-level model evaluated on random validation slices and battery holdouts",
           "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
           "delta_SP": DELTA_SP, "dataset_length": DATASET_LENGTH, "dsc_num": DSC_NUM,
           "torch": torch.__version__, "numpy": np.__version__}

    # ---- Build per-battery datasets (11 train batteries) ----
    print("Building train datasets ...", flush=True)
    t0 = time.time()
    train_data, train_targets = {}, {}
    for name in TRAIN_NAMES:
        X, y, Trange = dataset_generation_function(TRAIN_DATA_DIR, f"DSC_{name}", f"ARC_{name}",
                                                   DATASET_LENGTH, DSC_NUM, DELTA_SP)
        train_data[name] = torch.tensor(X, dtype=torch.float32)
        train_targets[name] = torch.tensor(y, dtype=torch.float32)
        print(f"  {name}: {len(y)} samples, Temp_range [{Trange[0]}, {Trange[-1]}]", flush=True)
    log["data_gen_seconds"] = round(time.time() - t0, 1)
    log["train_samples_per_battery"] = {k: len(v) for k, v in train_targets.items()}

    all_data = torch.cat([train_data[n] for n in TRAIN_NAMES], dim=0)
    all_targets = torch.cat([train_targets[n] for n in TRAIN_NAMES], dim=0)
    n_total = len(all_targets)
    log["n_train_battery_samples_total"] = n_total
    print(f"Total train-battery samples: {n_total}", flush=True)

    # ---- One fixed sample-level random 80/20 split ----
    set_seed(SEED)
    g_split = torch.Generator().manual_seed(SEED)
    dataset = TensorDataset(all_data, all_targets)
    train_size = int(0.8 * n_total)
    val_size = n_total - train_size
    r_train_ds, r_val_ds = random_split(dataset, [train_size, val_size], generator=g_split)
    log["train_size"], log["sample_val_size"] = train_size, val_size

    t0 = time.time()
    model_shared, train_loss, sample_val_loss = train_model(
        r_train_ds, r_val_ds, EPOCHS, LR, BATCH_SIZE, SEED, tag="shared"
    )
    log["train_seconds"] = round(time.time() - t0, 1)
    log["sample_val_mse_final"] = float(sample_val_loss[-1])
    log["sample_val_mse_best"] = float(min(sample_val_loss))
    log["sample_val_mse_best_epoch"] = int(np.argmin(sample_val_loss)) + 1
    torch.save(model_shared.state_dict(), os.path.join(HERE, "model_shared.pt"))

    # ---- Evaluate the same fitted weights on the 3 holdout batteries ----
    print("Evaluating on holdout test batteries ...", flush=True)
    test_results = {}
    for name in TEST_NAMES:
        X, y, Trange = dataset_generation_function(TEST_DATA_DIR, f"DSC_{name}", f"ARC_{name}",
                                                   DATASET_LENGTH, DSC_NUM, DELTA_SP)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = np.asarray(y, dtype=np.float32)
        pred = predict(model_shared, X_t)
        mse = float(np.mean((pred - y_t) ** 2))
        test_results[name] = {"mse": mse, "n_samples": int(len(y_t)),
                              "paper_table_S8": PAPER_TABLE_S8[name],
                              "temp_range": [float(Trange[0]), float(Trange[-1])]}
        print(f"  {name}: MSE = {mse:.4f} (paper Table S8: {PAPER_TABLE_S8[name]})", flush=True)

        # pred vs ground truth curve
        plt.figure(figsize=(10, 4))
        plt.plot(Trange, y_t, label="Ground truth log(dT/dt)", lw=1.5)
        plt.plot(Trange, pred, label="Predicted (shared model)", lw=1.2, alpha=0.8)
        plt.xlabel("Cutoff temperature (°C)")
        plt.ylabel("log(dT/dt)")
        plt.title(f"Holdout battery {name} — MSE {mse:.3f} (paper Table S8: {PAPER_TABLE_S8[name]})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, f"holdout_curve_{name}.png"), dpi=150)
        plt.close()

    log["holdout_test_mse_per_battery"] = test_results
    log["total_seconds"] = round(time.time() - t_start, 1)

    # ---- Figures: loss curves + MSE comparison bar chart ----
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, EPOCHS + 1), train_loss, label="train loss", alpha=0.8)
    plt.plot(range(1, EPOCHS + 1), sample_val_loss, label="sample-level validation loss", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.title(f"Fixed 80/20 split loss curves (seed={SEED}, {EPOCHS} epochs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "exp_b_loss_curves.png"), dpi=150)
    plt.close()

    labels = ["Random 20% validation\n(sample-level)"] + [f"{n}\n(battery holdout)" for n in TEST_NAMES]
    vals = [log["sample_val_mse_final"]] + [test_results[n]["mse"] for n in TEST_NAMES]
    paper_vals = [None] + [PAPER_TABLE_S8[n] for n in TEST_NAMES]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(labels, vals, color=["#4C9BD6", "#E07B54", "#E07B54", "#E07B54"])
    for i, p in enumerate(paper_vals):
        if p is not None:
            plt.hlines(p, i - 0.4, i + 0.4, colors="k", linestyles="dotted",
                       label="paper Table S8" if i == 1 else None)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("MSE (log(dT/dt))")
    plt.title("Sample-level random split vs battery-level holdout")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "exp_b_mse_comparison.png"), dpi=150)
    plt.close()

    with open(os.path.join(HERE, "experiment_b_metrics.json"), "w") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print("\n==== SUMMARY ====")
    print(f"Sample-level val MSE (final): {log['sample_val_mse_final']:.4f} | best: {log['sample_val_mse_best']:.4f} @epoch {log['sample_val_mse_best_epoch']}")
    for n in TEST_NAMES:
        r = test_results[n]
        print(f"Holdout {n}: MSE {r['mse']:.4f} (paper {r['paper_table_S8']})")
    print(f"Total runtime: {log['total_seconds']} s")
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
