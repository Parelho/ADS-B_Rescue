import csv
import math
import random
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

from torch.cuda.amp import (
    autocast,
    GradScaler
)

import simplekml

# =========================================================
# PERFORMANCE
# =========================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# =========================================================
# CONFIG
# =========================================================
INPUT_LEN = 32
FUTURE_LEN = 8

STRIDE = 4

BATCH_SIZE = 16384
EPOCHS = 25

LR = 1e-3

HIDDEN_SIZE = 128
NUM_LAYERS = 5

INPUT_SIZE = 11

MIN_FLIGHT_POINTS = (
    INPUT_LEN
    + FUTURE_LEN
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("DEVICE:", DEVICE)

# =========================================================
# EARTH
# =========================================================
EARTH_RADIUS = 6378137.0

# =========================================================
# LOAD DATA
# =========================================================
file_path = "trajectory.csv"

flights = defaultdict(list)

with open(file_path, newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        try:
            t = int(row["time"])

            icao24 = row["icao24"].strip()

            callsign = row["callsign"].strip()

            lat = float(row["lat"])
            lon = float(row["lon"])

            velocity = float(
                row["velocity"]
            )

            heading = float(
                row["heading"]
            )

            vertrate = float(
                row["vertrate"]
            )

            altitude = float(
                row["geoaltitude"]
            )

            key = (
                icao24,
                callsign
            )

            flights[key].append((
                t,
                lat,
                lon,
                velocity,
                heading,
                vertrate,
                altitude
            ))

        except Exception as e:

            print(
                "ROW ERROR:",
                e
            )

print(
    "Grouped flights:",
    len(flights)
)

# =========================================================
# SORT
# =========================================================
for key in flights:

    flights[key] = sorted(
        flights[key],
        key=lambda x: x[0]
    )

# =========================================================
# LATLON -> XY
# =========================================================
def latlon_to_xy(
    lat,
    lon,
    ref_lat,
    ref_lon
):

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    ref_lat_rad = math.radians(ref_lat)
    ref_lon_rad = math.radians(ref_lon)

    x = (
        (lon_rad - ref_lon_rad)
        * math.cos(ref_lat_rad)
        * EARTH_RADIUS
    )

    y = (
        (lat_rad - ref_lat_rad)
        * EARTH_RADIUS
    )

    return x, y

# =========================================================
# XY -> LATLON
# =========================================================
def xy_to_latlon(
    x,
    y,
    ref_lat,
    ref_lon
):

    lat = (
        ref_lat
        + np.degrees(
            y / EARTH_RADIUS
        )
    )

    lon = (
        ref_lon
        + np.degrees(
            x / (
                EARTH_RADIUS
                * np.cos(
                    np.radians(ref_lat)
                )
            )
        )
    )

    return lat, lon

# =========================================================
# BUILD WINDOWS
# =========================================================
samples = []

for key, flight in flights.items():

    if len(flight) < MIN_FLIGHT_POINTS:
        continue

    ref_lat = flight[0][1]
    ref_lon = flight[0][2]

    features = []

    prev_t = flight[0][0]

    prev_heading = flight[0][4]

    prev_vx = 0
    prev_vy = 0
    prev_vz = 0

    for (
        t,
        lat,
        lon,
        velocity,
        heading,
        vertrate,
        altitude
    ) in flight:

        # =====================================
        # POSITION
        # =====================================
        x, y = latlon_to_xy(
            lat,
            lon,
            ref_lat,
            ref_lon
        )

        # =====================================
        # TIME DELTA
        # =====================================
        dt = max(
            t - prev_t,
            1
        )

        prev_t = t

        # =====================================
        # VELOCITY
        # =====================================
        heading_rad = np.radians(
            heading
        )

        vx = (
            velocity
            * np.sin(heading_rad)
        )

        vy = (
            velocity
            * np.cos(heading_rad)
        )

        vz = vertrate

        # =====================================
        # ACCELERATION
        # =====================================
        ax = (
            vx - prev_vx
        ) / dt

        ay = (
            vy - prev_vy
        ) / dt

        az = (
            vz - prev_vz
        ) / dt

        prev_vx = vx
        prev_vy = vy
        prev_vz = vz

        # =====================================
        # TURN RATE
        # =====================================
        heading_diff = (
            heading
            - prev_heading
            + 180
        ) % 360 - 180

        turn_rate = (
            heading_diff / dt
        )

        prev_heading = heading

        # =====================================
        # FEATURES
        # =====================================
        features.append([

            # position
            x,
            y,
            altitude,

            # velocity
            vx,
            vy,
            vz,

            # acceleration
            ax,
            ay,
            az,

            # turn
            turn_rate,

            # timing
            dt
        ])

    features = np.array(
        features,
        dtype=np.float32
    )

    # =====================================
    # BUILD WINDOWS
    # =====================================
    max_start = (
        len(features)
        - INPUT_LEN
        - FUTURE_LEN
    )

    for start in range(
        0,
        max_start + 1,
        STRIDE
    ):

        x = features[
            start:
            start + INPUT_LEN
        ].copy()

        y = features[
            start + INPUT_LEN:
            start + INPUT_LEN + FUTURE_LEN
        ].copy()

        # =====================================
        # RELATIVE POSITIONING
        # =====================================
        anchor = x[-1].copy()

        x[:, :3] -= anchor[:3]
        y[:, :3] -= anchor[:3]

        samples.append((
            x,
            y,
            ref_lat,
            ref_lon,
            anchor
        ))

print("Samples:", len(samples))

if len(samples) == 0:

    raise ValueError(
        "No samples generated"
    )

# =========================================================
# NORMALIZATION
# =========================================================
all_inputs = np.concatenate([
    s[0]
    for s in samples
], axis=0)

feature_mean = all_inputs.mean(
    axis=0
)

feature_std = all_inputs.std(
    axis=0
)

feature_std[
    feature_std == 0
] = 1

# =========================================================
# NORMALIZE
# =========================================================
normalized_samples = []

for (
    x,
    y,
    ref_lat,
    ref_lon,
    anchor
) in samples:

    x = (
        x - feature_mean
    ) / feature_std

    y = (
        y - feature_mean
    ) / feature_std

    normalized_samples.append((
        x,
        y,
        ref_lat,
        ref_lon,
        anchor
    ))

samples = normalized_samples

# =========================================================
# SPLIT
# =========================================================
indices = list(
    range(len(samples))
)

random.shuffle(indices)

split = int(
    0.98 * len(indices)
)

train_idx = indices[:split]
test_idx = indices[split:]

# =========================================================
# BUILD ARRAYS
# =========================================================
train_inputs = np.array([
    samples[i][0]
    for i in train_idx
], dtype=np.float32)

train_targets = np.array([
    samples[i][1]
    for i in train_idx
], dtype=np.float32)

test_samples = [
    samples[i]
    for i in test_idx
]

print(
    "Training windows:",
    len(train_inputs)
)

# =========================================================
# DATASET
# =========================================================
class FlightDataset(Dataset):

    def __init__(
        self,
        inputs,
        targets
    ):

        self.inputs = torch.tensor(
            inputs,
            dtype=torch.float32
        )

        self.targets = torch.tensor(
            targets,
            dtype=torch.float32
        )

    def __len__(self):

        return len(
            self.inputs
        )

    def __getitem__(
        self,
        idx
    ):

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
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

# =========================================================
# MODEL
# =========================================================
class TrajectoryGRU(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        future_len,
        num_layers
    ):

        super().__init__()

        self.future_len = future_len

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # =====================================
        # ATTENTION
        # =====================================
        self.attn = nn.Sequential(

            nn.Linear(
                hidden_size,
                128
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1
            )
        )

        # =====================================
        # HEAD
        # =====================================
        self.head = nn.Sequential(

            nn.Linear(
                hidden_size,
                hidden_size
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                hidden_size,
                future_len
                * input_size
            )
        )

    def forward(
        self,
        x
    ):

        out, _ = self.gru(x)

        # =====================================
        # ATTENTION WEIGHTS
        # =====================================
        weights = self.attn(out)

        weights = torch.softmax(
            weights,
            dim=1
        )

        context = (
            out * weights
        ).sum(dim=1)

        pred = self.head(context)

        pred = pred.view(
            x.shape[0],
            self.future_len,
            x.shape[2]
        )

        return pred

model = TrajectoryGRU(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    future_len=FUTURE_LEN,
    num_layers=NUM_LAYERS
).to(DEVICE)

model = torch.compile(
    model
)

# =========================================================
# LOSS
# =========================================================
criterion = nn.SmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
)

scaler = GradScaler()

# =========================================================
# UNNORMALIZE
# =========================================================
def unnormalize(x):

    return (
        x * feature_std
    ) + feature_mean

# =========================================================
# TRAIN
# =========================================================
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

        # =====================================
        # NOISE AUGMENTATION
        # =====================================
        noise = (
            torch.randn_like(x)
            * 0.01
        )

        x_noisy = x + noise

        optimizer.zero_grad()

        with autocast():

            pred = model(
                x_noisy
            )

            # =================================
            # POSITION LOSS
            # =================================
            pos_loss = criterion(
                pred[:, :, :3],
                y[:, :, :3]
            )

            # =================================
            # VELOCITY LOSS
            # =================================
            vel_loss = criterion(
                pred[:, :, 3:6],
                y[:, :, 3:6]
            )

            # =================================
            # ACCELERATION LOSS
            # =================================
            accel_loss = criterion(
                pred[:, :, 6:9],
                y[:, :, 6:9]
            )

            # =================================
            # TURN LOSS
            # =================================
            turn_loss = criterion(
                pred[:, :, 9],
                y[:, :, 9]
            )

            # =================================
            # DIRECTION
            # =================================
            pred_dir = (
                pred[:, 1:, :2]
                - pred[:, :-1, :2]
            )

            real_dir = (
                y[:, 1:, :2]
                - y[:, :-1, :2]
            )

            curve_loss = criterion(
                pred_dir,
                real_dir
            )

            # =================================
            # SMOOTHNESS
            # =================================
            smooth_loss = criterion(

                pred[:, 1:, :3]
                - pred[:, :-1, :3],

                y[:, 1:, :3]
                - y[:, :-1, :3]
            )

            # =================================
            # FINAL
            # =================================
            loss = (

                pos_loss * 4.0

                + vel_loss * 2.0

                + accel_loss

                + turn_loss

                + curve_loss * 2.0

                + smooth_loss
            )

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        total_loss += loss.item()

    scheduler.step()

    avg_loss = (
        total_loss
        / len(train_loader)
    )

    print(
        f"Epoch "
        f"{epoch+1}/{EPOCHS} "
        f"Loss: {avg_loss:.6f}"
    )

    if avg_loss < best_loss:

        best_loss = avg_loss

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            "Saved best model:",
            best_loss
        )

# =========================================================
# LOAD BEST MODEL
# =========================================================
model.load_state_dict(
    torch.load(
        "best_model.pth"
    )
)

model.eval()

# =========================================================
# CLEAN COORDS
# =========================================================
def clean_coords(
    coords,
    eps=1e-7
):

    cleaned = []

    prev = None

    for lon, lat in coords:

        if (
            np.isnan(lat)
            or np.isnan(lon)
            or np.isinf(lat)
            or np.isinf(lon)
        ):
            continue

        point = (
            round(lon, 7),
            round(lat, 7)
        )

        if prev is not None:

            dlon = abs(
                point[0]
                - prev[0]
            )

            dlat = abs(
                point[1]
                - prev[1]
            )

            if (
                dlon < eps
                and dlat < eps
            ):
                continue

        cleaned.append(
            point
        )

        prev = point

    return cleaned

# =========================================================
# KML
# =========================================================
kml = simplekml.Kml()

for idx, sample in enumerate(
    test_samples[:100]
):

    (
        x_input,
        y_real,
        ref_lat,
        ref_lon,
        anchor
    ) = sample

    x_tensor = torch.tensor(
        x_input,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        with autocast():

            pred = model(
                x_tensor
            )

    pred = (
        pred
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # =====================================
    # UNNORMALIZE
    # =====================================
    x_input = unnormalize(
        x_input
    )

    y_real = unnormalize(
        y_real
    )

    pred = unnormalize(
        pred
    )

    # =====================================
    # RESTORE ANCHOR
    # =====================================
    x_input[:, :3] += anchor[:3]
    y_real[:, :3] += anchor[:3]
    pred[:, :3] += anchor[:3]

    # =====================================
    # INPUT
    # =====================================
    input_coords = []

    for row in x_input:

        lat, lon = xy_to_latlon(
            row[0],
            row[1],
            ref_lat,
            ref_lon
        )

        input_coords.append(
            (lon, lat)
        )

    # =====================================
    # REAL
    # =====================================
    real_coords = []

    for row in y_real:

        lat, lon = xy_to_latlon(
            row[0],
            row[1],
            ref_lat,
            ref_lon
        )

        real_coords.append(
            (lon, lat)
        )

    # =====================================
    # PRED
    # =====================================
    pred_coords = []

    for row in pred:

        lat, lon = xy_to_latlon(
            row[0],
            row[1],
            ref_lat,
            ref_lon
        )

        pred_coords.append(
            (lon, lat)
        )

    # =====================================
    # CLEAN
    # =====================================
    input_coords = clean_coords(
        input_coords
    )

    real_coords = clean_coords(
        real_coords
    )

    pred_coords = clean_coords(
        pred_coords
    )

    # =====================================
    # INPUT LINE
    # =====================================
    if len(input_coords) >= 2:

        line = kml.newlinestring(
            name=f"Input {idx}",
            coords=input_coords
        )

        line.style.linestyle.color = (
            simplekml.Color.blue
        )

        line.style.linestyle.width = 4

    # =====================================
    # REAL LINE
    # =====================================
    if len(real_coords) >= 2:

        line = kml.newlinestring(
            name=f"Real {idx}",
            coords=real_coords
        )

        line.style.linestyle.color = (
            simplekml.Color.green
        )

        line.style.linestyle.width = 4

    # =====================================
    # PREDICTED LINE
    # =====================================
    if len(pred_coords) >= 2:

        line = kml.newlinestring(
            name=f"Predicted {idx}",
            coords=pred_coords
        )

        line.style.linestyle.color = (
            simplekml.Color.red
        )

        line.style.linestyle.width = 4

# =========================================================
# SAVE
# =========================================================
kml.save("prediction.kml")

print("Saved prediction.kml")