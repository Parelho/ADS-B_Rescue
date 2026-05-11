import csv
import numpy as np
import random
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.cuda.amp import autocast, GradScaler

import simplekml

# =========================
# PERFORMANCE SETTINGS
# =========================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# =========================
# CONFIG
# =========================
SEQ_LEN = 128

INPUT_LEN = 8
FUTURE_LEN = 2

STRIDE = 1

BATCH_SIZE = 4096
EPOCHS = 200

LR = 1e-3
HIDDEN_SIZE = 128

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(DEVICE)

# =========================
# LOAD DATA
# =========================
file_path = "trajectory.csv"

flights = defaultdict(list)

with open(file_path, newline='') as f:

    reader = csv.DictReader(f)

    for row in reader:

        fs = int(row["firstseen"])

        dep = row["estdepartureairport"]
        arr = row["estarrivalairport"]

        key = (fs, dep, arr)

        t = int(row["time"])

        lat = float(row["lat"])
        lon = float(row["lon"])

        heading = float(row["heading"])

        flights[key].append(
            (t, lat, lon, heading)
        )

# =========================
# SORT FLIGHTS
# =========================
for key in flights:

    flights[key] = sorted(
        flights[key],
        key=lambda x: x[0]
    )

# =========================
# HELPERS
# =========================
def resample_path(path, target_len):

    x_old = np.linspace(0, 1, len(path))
    x_new = np.linspace(0, 1, target_len)

    resampled = np.zeros(
        (target_len, path.shape[1]),
        dtype=np.float32
    )

    for i in range(path.shape[1]):

        resampled[:, i] = np.interp(
            x_new,
            x_old,
            path[:, i]
        )

    return resampled

# =========================
# BUILD TRAJECTORIES
# =========================
all_paths = []
flight_refs = []

for key, flight in flights.items():

    if len(flight) < 50:
        continue

    coords = []

    prev_lat = flight[0][1]
    prev_lon = flight[0][2]

    for t, lat, lon, heading in flight:

        dlat = lat - prev_lat
        dlon = lon - prev_lon

        coords.append([
            dlat,
            dlon,
            np.sin(np.radians(heading)),
            np.cos(np.radians(heading))
        ])

        prev_lat = lat
        prev_lon = lon

    coords = np.array(
        coords,
        dtype=np.float32
    )

    path = resample_path(
        coords,
        SEQ_LEN
    )

    all_paths.append(path)

    flight_refs.append(flight)

all_paths = np.array(
    all_paths,
    dtype=np.float32
)

all_paths = np.ascontiguousarray(all_paths)

# =========================
# NORMALIZATION
# =========================
feature_mean = all_paths.mean(axis=(0, 1))
feature_std = all_paths.std(axis=(0, 1))

feature_std[feature_std == 0] = 1

all_paths = (
    all_paths - feature_mean
) / feature_std

# =========================
# TRAIN / TEST SPLIT
# =========================
indices = list(range(len(all_paths)))

random.shuffle(indices)

split = int(0.99 * len(indices))

train_idx = indices[:split]
test_idx = indices[split:]

train_paths = all_paths[train_idx]
test_paths = all_paths[test_idx]

test_flights = [
    flight_refs[i]
    for i in test_idx
]

# =========================
# CREATE SLIDING WINDOWS
# =========================
train_inputs = []
train_targets = []

for path in train_paths:

    max_start = (
        SEQ_LEN
        - INPUT_LEN
        - FUTURE_LEN
    )

    for start in range(
        0,
        max_start + 1,
        STRIDE
    ):

        x = path[
            start:
            start + INPUT_LEN
        ]

        y = path[
            start + INPUT_LEN:
            start + INPUT_LEN + FUTURE_LEN
        ]

        train_inputs.append(x)
        train_targets.append(y)

train_inputs = torch.tensor(
    np.array(train_inputs),
    dtype=torch.float32
)

train_targets = torch.tensor(
    np.array(train_targets),
    dtype=torch.float32
)

print("Training windows:", len(train_inputs))

# =========================
# DATASET
# =========================
class FlightDataset(Dataset):

    def __init__(
        self,
        inputs,
        targets
    ):

        self.inputs = inputs
        self.targets = targets

    def __len__(self):

        return len(self.inputs)

    def __getitem__(self, idx):

        return (
            self.inputs[idx],
            self.targets[idx]
        )

train_dataset = FlightDataset(
    train_inputs,
    train_targets
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True
)

# =========================
# MODEL
# =========================
class GRUPredictor(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=128,
        future_len=2
    ):

        super().__init__()

        self.future_len = future_len
        self.input_size = input_size

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            batch_first=True,
            num_layers=1
        )

        self.fc = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_size,
                future_len * input_size
            )
        )

    def forward(self, x):

        _, hidden = self.gru(x)

        hidden = hidden[-1]

        out = self.fc(hidden)

        out = out.view(
            -1,
            self.future_len,
            self.input_size
        )

        return out

model = GRUPredictor(
    input_size=4,
    hidden_size=HIDDEN_SIZE,
    future_len=FUTURE_LEN
).to(DEVICE)

model = torch.compile(model)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)

scaler = GradScaler()

# =========================
# TRAIN
# =========================
best_loss = float("inf")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for x, y in train_loader:

        x = x.to(
            DEVICE,
            non_blocking=True
        )

        y = y.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        with autocast():

            pred = model(x)

            loss = criterion(
                pred,
                y
            )

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.6f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        print(
            f"New best model saved! "
            f"Loss: {best_loss:.6f}"
        )

# =========================
# DENORMALIZATION
# =========================
def unnormalize(path):

    return (
        path * feature_std
    ) + feature_mean

def denorm(path, lat0, lon0):

    coords = []

    current_lat = lat0
    current_lon = lon0

    for step in path:

        current_lat += step[0]
        current_lon += step[1]

        coords.append([
            current_lat,
            current_lon
        ])

    return np.array(coords)

# =========================
# PREDICT TEST FLIGHTS
# =========================
model.eval()

kml = simplekml.Kml()

for idx, (sample, flight) in enumerate(
    zip(test_paths, test_flights)
):

    print(
        f"Predicting "
        f"{idx+1}/{len(test_paths)}"
    )

    lat0 = flight[0][1]
    lon0 = flight[0][2]

    # =========================
    # UNNORMALIZE REAL PATH
    # =========================
    real_full = unnormalize(sample)

    # =========================
    # INITIAL INPUT
    # =========================
    current_window = sample[:INPUT_LEN].copy()

    # Keep COMPLETE trajectory
    reconstructed = list(
        real_full[:INPUT_LEN]
    )

    total_future = (
        SEQ_LEN - INPUT_LEN
    )

    steps_needed = int(np.ceil(
        total_future / FUTURE_LEN
    ))

    # =========================
    # AUTOREGRESSIVE LOOP
    # =========================
    for _ in range(steps_needed):

        x_tensor = torch.tensor(
            current_window,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():

            with autocast():

                pred = model(x_tensor)

        pred = (
            pred.squeeze(0)
            .cpu()
            .numpy()
        )

        # =========================
        # UNNORMALIZE PREDICTION
        # =========================
        pred_un = unnormalize(pred)

        # Save prediction
        reconstructed.extend(pred_un)

        # =========================
        # FEEDBACK INTO MODEL
        # =========================
        pred_norm = (
            pred_un - feature_mean
        ) / feature_std

        current_window = np.concatenate([
            current_window[FUTURE_LEN:],
            pred_norm
        ])

    # =========================
    # FINAL TRAJECTORY
    # =========================
    reconstructed = np.array(
        reconstructed[:SEQ_LEN]
    )

    # =========================
    # CONVERT DELTAS -> GPS
    # =========================
    real_coords = denorm(
        real_full[:, :2],
        lat0,
        lon0
    )

    pred_coords = denorm(
        reconstructed[:, :2],
        lat0,
        lon0
    )

    input_coords = denorm(
        real_full[:INPUT_LEN, :2],
        lat0,
        lon0
    )

    # =========================
    # KML
    # =========================

    # INPUT
    partial_line = kml.newlinestring(
        name=f"Input {idx}",
        coords=[
            (p[1], p[0])
            for p in input_coords
        ]
    )

    partial_line.style.linestyle.color = (
        simplekml.Color.blue
    )

    partial_line.style.linestyle.width = 4

    # REAL
    real_line = kml.newlinestring(
        name=f"Real {idx}",
        coords=[
            (p[1], p[0])
            for p in real_coords
        ]
    )

    real_line.style.linestyle.color = (
        simplekml.Color.green
    )

    real_line.style.linestyle.width = 4

    # PREDICTED
    pred_line = kml.newlinestring(
        name=f"Predicted {idx}",
        coords=[
            (p[1], p[0])
            for p in pred_coords
        ]
    )

    pred_line.style.linestyle.color = (
        simplekml.Color.red
    )

    pred_line.style.linestyle.width = 4

# =========================
# SAVE
# =========================
kml.save("prediction.kml")

print("Saved prediction.kml")

# Force clean exit
import os
os._exit(0)