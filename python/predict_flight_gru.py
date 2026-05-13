import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# =========================================================
# Reproducibility
# =========================================================

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================================================
# Config
# =========================================================

FEATURE_COLS = ["lat", "lon", "altitude"]
TARGET_COLS = ["lat", "lon", "altitude"]

# Lat/lon matter most. Altitude is only a small helper signal.
TARGET_WEIGHTS = np.array([0.33, 0.33, 0.34], dtype=np.float32)

# Flight identity: one flight is identified by the firstseen timestamp
# and the airport pair.
KEY_COLS = ["firstseen", "estdepartureairport", "estarrivalairport"]

TIME_CANDIDATES = ["time", "timestamp", "lastseen", "firstseen", "time_position"]


@dataclass
class FlightSample:
    key: Tuple[str, str, str]
    values: np.ndarray  # (T, 3) float32 in original units
    times: Optional[np.ndarray] = None


# =========================================================
# Data loading / cleaning
# =========================================================

def load_adsb_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()

    required = set(KEY_COLS + FEATURE_COLS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    # Make the numeric columns numeric.
    for col in FEATURE_COLS + ["firstseen", "time"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by flight id + time so each flight becomes an ordered trajectory.
    sort_cols = KEY_COLS + (["time"] if "time" in df.columns else [])
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    df = df.dropna(subset=KEY_COLS + FEATURE_COLS).reset_index(drop=True)

    print(f"[data] Loaded {len(df):,} rows from '{csv_path}'")
    return df


def split_flights(
    df: pd.DataFrame,
    val_ratio: float = 0.14,
    test_ratio: float = 0.01,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    flight_keys = df.groupby(KEY_COLS, sort=False).size().index.tolist()

    train_keys, temp_keys = train_test_split(
        flight_keys,
        test_size=val_ratio + test_ratio,
        random_state=seed,
    )

    rel_test = test_ratio / (val_ratio + test_ratio)
    val_keys, test_keys = train_test_split(
        temp_keys,
        test_size=rel_test,
        random_state=seed,
    )

    def _subset(keys: Sequence[Tuple[str, str, str]]) -> pd.DataFrame:
        mask = df.set_index(KEY_COLS).index.isin(keys)
        return df.loc[mask].copy().reset_index(drop=True)

    train_df = _subset(train_keys)
    val_df = _subset(val_keys)
    test_df = _subset(test_keys)

    print(
        f"[split] Train flights={len(train_keys)} Val flights={len(val_keys)} Test flights={len(test_keys)}"
    )
    return train_df, val_df, test_df


def group_flights(df: pd.DataFrame) -> List[FlightSample]:
    flights: List[FlightSample] = []
    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)

    for key, grp in df.groupby(KEY_COLS, sort=False):
        if len(grp) < 6:
            continue
        if time_col is not None:
            grp = grp.sort_values(time_col, kind="mergesort")

        values = grp[FEATURE_COLS].to_numpy(dtype=np.float32)
        times = grp[time_col].to_numpy(dtype=np.float32) if time_col else None
        key = tuple(str(v) for v in key)
        flights.append(FlightSample(key, values, times))

    return flights


# =========================================================
# Scaler
# =========================================================

def fit_scaler(flights: List[FlightSample]) -> StandardScaler:
    scaler = StandardScaler()
    all_points = np.concatenate([f.values for f in flights], axis=0)
    scaler.fit(all_points)
    return scaler


# =========================================================
# Dataset and collate
# =========================================================

class FlightSequenceDataset(Dataset):
    def __init__(self, flights: List[FlightSample], scaler: StandardScaler) -> None:
        self.flights = flights
        self.scaler = scaler

    def __len__(self) -> int:
        return len(self.flights)

    def __getitem__(self, idx: int):
        sample = self.flights[idx]
        original = sample.values.astype(np.float32)
        scaled = self.scaler.transform(original).astype(np.float32)

        return {
            "x": torch.tensor(scaled, dtype=torch.float32),
            "y": torch.tensor(scaled, dtype=torch.float32),
            "length": len(scaled),
            "key": sample.key,
        }


def collate_flights(batch):
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    max_len = int(lengths.max().item())

    x_pad = torch.zeros(len(batch), max_len, 3, dtype=torch.float32)
    y_pad = torch.zeros(len(batch), max_len, 3, dtype=torch.float32)
    valid_pad = torch.zeros(len(batch), max_len, dtype=torch.bool)
    keys = []

    for i, item in enumerate(batch):
        T = item["length"]
        x_pad[i, :T] = item["x"]
        y_pad[i, :T] = item["y"]
        valid_pad[i, :T] = True
        keys.append(item["key"])

    return {
        "x": x_pad,
        "y": y_pad,
        "valid_mask": valid_pad,
        "lengths": lengths,
        "keys": keys,
    }


# =========================================================
# Model
# =========================================================

class FlightSequenceModel(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=3,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return self.head(out)


# =========================================================
# Loss / metrics
# =========================================================

def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    target_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Weighted MSE with lat/lon emphasized over altitude."""
    valid = valid_mask.unsqueeze(-1).float()
    mse = (pred - target) ** 2

    if target_weights is None:
        target_weights = torch.tensor(TARGET_WEIGHTS, device=pred.device, dtype=pred.dtype)
    else:
        target_weights = target_weights.to(device=pred.device, dtype=pred.dtype)

    mse = mse * target_weights.view(1, 1, -1)
    loss = (mse * valid).sum() / valid.sum().clamp_min(1.0)
    return loss


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        lengths = batch["lengths"].to(device)

        pred = model(x, lengths)
        loss = weighted_mse_loss(pred, y, valid_mask)
        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / max(len(loader.dataset), 1)
    return avg_loss


@torch.no_grad()
def evaluate_in_original_units(model, loader, device, scaler: StandardScaler):
    model.eval()
    preds_all = []
    targets_all = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)

        pred = model(x, lengths)
        preds_all.append(pred.cpu().numpy())
        targets_all.append(y.cpu().numpy())

    pred = np.concatenate(preds_all, axis=0)
    true = np.concatenate(targets_all, axis=0)

    n, t, c = pred.shape
    pred_inv = scaler.inverse_transform(pred.reshape(-1, c)).reshape(n, t, c)
    true_inv = scaler.inverse_transform(true.reshape(-1, c)).reshape(n, t, c)

    diff = pred_inv - true_inv
    rmse = math.sqrt(float((diff ** 2).mean()))
    mae = float(np.abs(diff).mean())
    horiz_rmse = math.sqrt(float((diff[..., :2] ** 2).mean()))

    return {
        "rmse_all": rmse,
        "mae_all": mae,
        "horiz_rmse": horiz_rmse,
        "pred_inv": pred_inv,
        "true_inv": true_inv,
    }


# =========================================================
# Train / predict helpers
# =========================================================

def make_loaders(train_flights, val_flights, test_flights, scaler, batch_size: int):
    train_ds = FlightSequenceDataset(train_flights, scaler)
    val_ds = FlightSequenceDataset(val_flights, scaler)
    test_ds = FlightSequenceDataset(test_flights, scaler)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_flights)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_flights)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_flights)
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, device, epochs: int, lr: float):
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min", factor=0.5, patience=3)

    best_val = float("inf")
    best_state = None
    train_hist = []
    val_hist = []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0

        for batch in train_loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            lengths = batch["lengths"].to(device)

            pred = model(x, lengths)
            loss = weighted_mse_loss(pred, y, valid_mask)

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total += loss.item() * x.size(0)

        train_loss = total / max(len(train_loader.dataset), 1)
        val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)

        train_hist.append(train_loss)
        val_hist.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch {epoch:>3}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"[train] Best val loss: {best_val:.6f}")
    return model, train_hist, val_hist


@torch.no_grad()
def predict_flight(model, scaler, sample: FlightSample, device):
    original = sample.values.astype(np.float32)
    scaled = scaler.transform(original).astype(np.float32)

    x = torch.tensor(scaled[None], dtype=torch.float32, device=device)
    lengths = torch.tensor([len(scaled)], dtype=torch.long, device=device)

    pred = model(x, lengths).cpu().numpy()[0]
    pred_inv = scaler.inverse_transform(pred)
    return original, pred_inv


# =========================================================
# Main
# =========================================================

def run(
    csv_path: str,
    epochs: int = 25,
    batch_size: int = 32,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    seed: int = 42,
    save_path: str = "adsb_sequence_model.pt",
    load_path: Optional[str] = None,
    plot_path: Optional[str] = "sequence_example.png",
):
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE: {device}")

    df = load_adsb_csv(csv_path)
    train_df, val_df, test_df = split_flights(df, seed=seed)

    train_flights = group_flights(train_df)
    val_flights = group_flights(val_df)
    test_flights = group_flights(test_df)

    print(f"[data] Train flights kept: {len(train_flights)}")
    print(f"[data] Val flights kept:   {len(val_flights)}")
    print(f"[data] Test flights kept:  {len(test_flights)}")

    scaler = fit_scaler(train_flights)
    train_loader, val_loader, test_loader = make_loaders(
        train_flights, val_flights, test_flights, scaler, batch_size
    )

    model = FlightSequenceModel(hidden_dim=hidden_dim).to(device)

    if load_path and os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        scaler.mean_ = checkpoint["scaler_mean"]
        scaler.scale_ = checkpoint["scaler_scale"]
        scaler.var_ = checkpoint["scaler_var"]
        scaler.n_features_in_ = 3
        print(f"[load] Loaded checkpoint from '{load_path}'")
    else:
        model, train_hist, val_hist = train_model(model, train_loader, val_loader, device, epochs, lr)
        torch.save(
            {
                "model_state": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "scaler_var": scaler.var_,
                "seed": seed,
                "hidden_dim": hidden_dim,
            },
            save_path,
        )
        print(f"[save] Saved checkpoint to '{save_path}'")

    test_stats = evaluate_in_original_units(model, test_loader, device, scaler)
    print(
        "[eval] "
        f"HorizRMSE={test_stats['horiz_rmse']:.3f}  "
        f"RMSE={test_stats['rmse_all']:.3f}  "
        f"MAE={test_stats['mae_all']:.3f}"
    )
    print("[note] This version trains on continuous paths with no synthetic gaps.")

    if plot_path and len(test_flights) > 0:
        import matplotlib.pyplot as plt

        sample = random.choice(test_flights)
        original, predicted = predict_flight(model, scaler, sample, device)

        fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
        titles = ["Latitude", "Longitude", "Altitude"]
        for i, ax in enumerate(axes):
            ax.plot(original[:, i], label="true")
            ax.plot(predicted[:, i], label="predicted", linestyle="--")
            ax.set_title(titles[i])
            ax.grid(alpha=0.3)
        axes[0].legend()
        axes[-1].set_xlabel("time step")
        fig.suptitle("Continuous flight path example")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=160, bbox_inches="tight")
        print(f"[plot] Saved → '{plot_path}'")

    return model, scaler, test_stats


# =========================================================
# CLI
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Simple ADS-B continuous-path model")
    parser.add_argument("--csv", type=str, default="trajectory.csv", help="Path to ADS-B CSV")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default="adsb_sequence_model.pt")
    parser.add_argument("--load-path", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default="sequence_example.png")
    args = parser.parse_args()

    run(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        seed=args.seed,
        save_path=args.save_path,
        load_path=args.load_path,
        plot_path=args.plot_path,
    )


if __name__ == "__main__":
    main()
