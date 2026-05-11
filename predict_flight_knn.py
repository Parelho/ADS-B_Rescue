import csv
import numpy as np
import random
from collections import defaultdict
import simplekml

file_path = "trajectory.csv"

# Group by:
# (firstseen, departure airport, arrival airport)
flights = defaultdict(list)

# ===== LOAD DATA =====
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
        altitude = float(row["altitude"])
        heading = float(row["heading"])

        flights[key].append((t, lat, lon, altitude, heading))

# Sort each flight by time
for key in flights:
    flights[key] = sorted(flights[key], key=lambda x: x[0])

# ===== HELPERS =====
def resample_path(path, target_len):
    x_old = np.linspace(0, 1, len(path))
    x_new = np.linspace(0, 1, target_len)

    resampled = np.zeros((target_len, path.shape[1]))

    for i in range(path.shape[1]):
        resampled[:, i] = np.interp(x_new, x_old, path[:, i])

    return resampled


def denorm(path, lat0, lon0, alt0):
    lat = path[:, 0] + lat0
    lon = path[:, 1] + lon0
    alt = path[:, 2] + alt0

    return np.stack([lat, lon, alt], axis=1)

# ===== BUILD DATASET =====
SEQ_LEN = 200

X_seq = []
valid_flights = []
flight_metadata = []

for key, flight in flights.items():
    fs, dep, arr = key

    if len(flight) < 200:
        continue

    lat0 = flight[0][1]
    lon0 = flight[0][2]
    alt0 = flight[0][3]

    coords = []

    for t, lat, lon, altitude, heading in flight:
        coords.append([
            lat - lat0,
            lon - lon0,
            altitude - alt0,
            np.sin(np.radians(heading)),
            np.cos(np.radians(heading))
        ])

    coords = np.array(coords)

    path = resample_path(coords, SEQ_LEN)

    X_seq.append(path)

    valid_flights.append((key, flight))

    # Save airport pair metadata
    flight_metadata.append((dep, arr))

X_seq = np.array(X_seq)

# ===== TRAIN / TEST SPLIT =====
indices = list(range(len(X_seq)))
random.shuffle(indices)

split = int(0.99 * len(indices))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X_seq[train_idx]
X_test = X_seq[test_idx]

test_flights = [valid_flights[i] for i in test_idx]

# Airport metadata
train_airports = [flight_metadata[i] for i in train_idx]
test_airports = [flight_metadata[i] for i in test_idx]

# ===== FLATTEN TRAIN TRAJECTORIES =====
train_features = X_train.reshape(X_train.shape[0], -1)

# ===== MASKED DISTANCE =====
def masked_distance(a, b, known_len):
    a = a.reshape(SEQ_LEN, 5)
    b = b.reshape(SEQ_LEN, 5)

    return np.linalg.norm(a[:known_len] - b[:known_len])

# ===== PREDICT FUNCTION =====
def predict_knn(partial_path, dep_airport, arr_airport):
    partial_len = len(partial_path)

    padded = np.zeros((SEQ_LEN, 5))
    padded[:partial_len] = partial_path

    query = padded.reshape(-1)

    dists = []

    for i, traj in enumerate(train_features):

        # Only compare flights with same airports
        train_dep, train_arr = train_airports[i]

        if train_dep != dep_airport:
            continue

        if train_arr != arr_airport:
            continue

        d = masked_distance(query, traj, partial_len)

        dists.append((i, d))

    dists.sort(key=lambda x: x[1])

    best_idx = dists[0][0]

    best_match = X_train[best_idx].copy()

    # Preserve known part
    best_match[:partial_len] = partial_path

    return best_match

# ===== KML OUTPUT =====
kml = simplekml.Kml()

for idx, ((key, flight), airport_pair) in enumerate(zip(test_flights, test_airports)):

    print(f"Processing flight {idx+1}/{len(test_flights)}")

    fs, dep, arr = key

    lat0 = flight[0][1]
    lon0 = flight[0][2]
    alt0 = flight[0][3]

    coords = []

    for t, lat, lon, altitude, h in flight:
        coords.append([
            lat - lat0,
            lon - lon0,
            altitude - alt0,
            np.sin(np.radians(h)),
            np.cos(np.radians(h))
        ])

    coords = np.array(coords)

    path = resample_path(coords, SEQ_LEN)

    # ===== PARTIAL INPUT =====
    partial_len = random.randint(int(SEQ_LEN * 0.3), int(SEQ_LEN * 0.9))

    partial = path[:partial_len]

    predicted = predict_knn(partial, dep, arr)

    real = denorm(path[:, :3], lat0, lon0, alt0)
    pred = denorm(predicted[:, :3], lat0, lon0, alt0)

    # ===== REAL PATH =====
    real_line = kml.newlinestring(
        name=f"Real {idx} {dep}->{arr}",
        coords=[(p[1], p[0], p[2]) for p in real]
    )

    real_line.altitudemode = simplekml.AltitudeMode.absolute
    real_line.extrude = 1
    real_line.style.linestyle.color = simplekml.Color.green

    # ===== PREDICTED PATH =====
    pred_line = kml.newlinestring(
        name=f"Pred {idx} {dep}->{arr}",
        coords=[(p[1], p[0], p[2]) for p in pred]
    )

    pred_line.altitudemode = simplekml.AltitudeMode.absolute
    pred_line.extrude = 1
    pred_line.style.linestyle.color = simplekml.Color.red

# ===== SAVE =====
kml.save("knn_flight_predictions.kml")

print("Saved knn_flight_predictions.kml")