import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import simplekml
import random
from collections import defaultdict

# ===== DEVICE =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

file_path = "trajectory.csv"
flights = defaultdict(list)

# ===== LOAD DATA =====
with open(file_path, newline='') as f:
  reader = csv.DictReader(f)
  for row in reader:
    fs = int(row["firstseen"])
    t = int(row["time"])
    lat = float(row["lat"])
    lon = float(row["lon"])
    heading = float(row["heading"])
    flights[fs].append((t, lat, lon, heading))

for fs in flights:
  flights[fs] = sorted(flights[fs], key=lambda x: x[0])

# ===== HELPERS =====
def resample_path(path, target_len):
  x_old = np.linspace(0, 1, len(path))
  x_new = np.linspace(0, 1, target_len)
  resampled = np.zeros((target_len, path.shape[1]))
  for i in range(path.shape[1]):
    resampled[:, i] = np.interp(x_new, x_old, path[:, i])
  return resampled

def recompute_velocity_and_heading(path):
  new_path = path.copy()
  for i in range(1, len(path)):
    dx = path[i, 0] - path[i-1, 0]
    dy = path[i, 1] - path[i-1, 1]

    new_path[i, 4] = dx
    new_path[i, 5] = dy

    angle = np.arctan2(dy, dx)
    new_path[i, 2] = np.sin(angle)
    new_path[i, 3] = np.cos(angle)

  new_path[0, 4:6] = 0
  return new_path

SEQ_LEN = 200

X_seq = []
X_vec = []
y_seq = []
valid_flights = []

# ===== DATASET =====
for fs, flight in flights.items():
  if len(flight) < 200:
    continue

  lat0 = flight[0][1]
  lon0 = flight[0][2]

  coords_list = []
  prev_lat, prev_lon, prev_t = None, None, None

  for t, lat, lon, heading in flight:
    lat_off = lat - lat0
    lon_off = lon - lon0

    if prev_lat is None:
      dx, dy = 0, 0
      dt = 0.0
    else:
      dx = lat - prev_lat
      dy = lon - prev_lon
      dt = t - prev_t

    dt = min(dt, 60) / 60.0

    coords_list.append([
      lat_off, lon_off,
      np.sin(np.radians(heading)),
      np.cos(np.radians(heading)),
      dx, dy,
      dt
    ])

    prev_lat, prev_lon, prev_t = lat, lon, t

  coords = np.array(coords_list)

  path = resample_path(coords, SEQ_LEN)
  path = recompute_velocity_and_heading(path)

  start = path[0][:2]
  end = path[-1][:2]

  X_seq.append(path[:-1])
  X_vec.append([start[0], start[1], end[0], end[1]])

  deltas = []
  for i in range(1, len(path)):
    dx = path[i, 0] - path[i-1, 0]
    dy = path[i, 1] - path[i-1, 1]
    deltas.append([dx, dy])

  y_seq.append(deltas)
  valid_flights.append((fs, flight))

X_seq = torch.tensor(np.array(X_seq), dtype=torch.float32).to(device)
X_vec = torch.tensor(np.array(X_vec), dtype=torch.float32).to(device)
y_seq = torch.tensor(np.array(y_seq), dtype=torch.float32).to(device)

# ===== TRAIN / VAL SPLIT =====
split = int(0.7 * X_seq.size(0))

X_train, X_val = X_seq[:split], X_seq[split:]
V_train, V_val = X_vec[:split], X_vec[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

# ===== MODEL =====
class TrajectoryModel(nn.Module):
  def __init__(self):
    super().__init__()

    self.input_proj = nn.Linear(7, 128)

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=128,
      nhead=4,
      dim_feedforward=128,
      batch_first=True
    )
    self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

    self.lstm = nn.LSTM(128 + 4, 128, batch_first=True)
    self.fc1 = nn.Linear(128, 64)
    self.fc2 = nn.Linear(64, 2)

  def forward(self, seq, vec):
    x = self.input_proj(seq)
    x = self.transformer(x)

    vec_rep = vec.unsqueeze(1).repeat(1, x.shape[1], 1)
    x = torch.cat([x, vec_rep], dim=-1)

    x, _ = self.lstm(x)
    x = torch.relu(self.fc1(x))
    return self.fc2(x)

model = TrajectoryModel().to(device)

# ===== LOSS =====
def loss_fn(y_pred, y_true):
  mse = torch.mean((y_pred - y_true) ** 2)

  pos_pred = torch.cumsum(y_pred, dim=1)
  pos_true = torch.cumsum(y_true, dim=1)

  endpoint_loss = torch.mean((pos_pred[:, -1] - pos_true[:, -1]) ** 2)

  return mse + 3.0 * endpoint_loss

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ===== TRAIN =====
epochs = 5000
batch_size = 128

best_loss = float("inf")
best_model = None
patience = 500
counter = 0

for epoch in range(epochs):
  model.train()

  perm = torch.randperm(X_train.size(0))

  for i in range(0, X_train.size(0), batch_size):
    idx = perm[i:i+batch_size]

    seq_batch = X_train[idx]
    vec_batch = V_train[idx]
    y_batch = y_train[idx]

    optimizer.zero_grad()
    pred = model(seq_batch, vec_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()
    optimizer.step()

  model.eval()
  with torch.no_grad():
    val_pred = model(X_val, V_val)
    val_loss = loss_fn(val_pred, y_val)

  if val_loss < best_loss:
    best_loss = val_loss
    best_model = model.state_dict()
    counter = 0
  else:
    counter += 1

  if epoch % 10 == 0:
    print(f"Epoch {epoch} | Train: {loss.item():.6f} | Val: {val_loss.item():.6f}")

  if counter >= patience:
    print("Early stopping triggered")
    break

model.load_state_dict(best_model)

# ===== DENORM =====
def denorm(path, lat0, lon0):
  lat = path[:, 0] + lat0
  lon = path[:, 1] + lon0
  heading = np.degrees(np.arctan2(path[:,2], path[:,3])) % 360
  return np.stack([lat, lon, heading], axis=1)

# ===== TEST 10 RANDOM FLIGHTS =====
kml = simplekml.Kml()

samples = random.sample(valid_flights, 10)

for idx, (fs, flight) in enumerate(samples):
  lat0, lon0 = flight[0][1], flight[0][2]

  coords = []
  prev_lat, prev_lon, prev_t = None, None, None

  for t, lat, lon, h in flight:
    lat_off, lon_off = lat-lat0, lon-lon0

    if prev_lat is None:
      dx, dy = 0, 0
      dt = 0.0
    else:
      dx = lat-prev_lat
      dy = lon-prev_lon
      dt = t - prev_t

    dt = min(dt, 60) / 60.0

    coords.append([
      lat_off, lon_off,
      np.sin(np.radians(h)),
      np.cos(np.radians(h)),
      dx, dy,
      dt
    ])

    prev_lat, prev_lon, prev_t = lat, lon, t

  coords = np.array(coords)

  path = resample_path(coords, SEQ_LEN)
  path = recompute_velocity_and_heading(path)

  start = path[0][:2]
  end = path[-1][:2]

  with torch.no_grad():
    pred_deltas = model(
      torch.tensor(path[:-1], dtype=torch.float32).unsqueeze(0).to(device),
      torch.tensor([[start[0], start[1], end[0], end[1]]], dtype=torch.float32).to(device)
    )[0].cpu().numpy()

  generated = []
  current = path[0].copy()

  for dx, dy in pred_deltas:
    new = current.copy()
    new[0] += dx
    new[1] += dy

    new[4] = dx
    new[5] = dy

    angle = np.arctan2(dy, dx)
    new[2] = np.sin(angle)
    new[3] = np.cos(angle)

    generated.append(new.copy())
    current = new

  predicted_norm = np.array(generated)

  real_path = denorm(path, lat0, lon0)
  predicted_real = denorm(predicted_norm, lat0, lon0)

  real_line = kml.newlinestring(
    name=f"Real {idx}",
    coords=[(p[1], p[0]) for p in real_path]
  )
  real_line.style.linestyle.color = simplekml.Color.green

  pred_line = kml.newlinestring(
    name=f"Pred {idx}",
    coords=[(p[1], p[0]) for p in predicted_real]
  )
  pred_line.style.linestyle.color = simplekml.Color.red

kml.save("flight_batch_test.kml")