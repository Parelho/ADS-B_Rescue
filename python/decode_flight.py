import csv
import re
import sys
csv.field_size_limit(sys.maxsize)

file_path = "output_old_knn.csv"

rows = []

with open(file_path, newline='') as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["estdepartureairport"] == "KMMU":
            rows.append(row)

track_pattern = re.compile(
    r'time=([0-9]+), latitude=([\-0-9.]+), longitude=([\-0-9.]+), altitude=([\-0-9.]+).*?heading=([\-0-9.]+)'
)

def parse_track(track_str):
    coords = []

    matches = track_pattern.findall(track_str)

    for t, lat, lon, heading, altitude in matches:
        coords.append(
            (
                int(t),
                float(lat),
                float(lon),
                float(heading),
                float(altitude)
            )
        )

    return coords

output_rows = []

for row in rows:
    track = row["track"]

    if not track:
        continue

    firstseen = row["firstseen"]

    estdepartureairport = row["estdepartureairport"]
    estarrivalairport = row["estarrivalairport"]

    coords = parse_track(track)

    for t, lat, lon, heading, altitude in coords:
        output_rows.append({
            "firstseen": int(firstseen),
            "time": t,
            "lat": lat,
            "lon": lon,
            "heading": heading,
            "altitude": altitude,
            "estdepartureairport": estdepartureairport,
            "estarrivalairport": estarrivalairport
        })

output_rows = sorted(
    output_rows,
    key=lambda x: (x["firstseen"], x["time"])
)

with open("trajectory.csv", "w", newline='') as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "firstseen",
            "time",
            "lat",
            "lon",
            "heading",
            "altitude",
            "estdepartureairport",
            "estarrivalairport"
        ]
    )

    writer.writeheader()
    writer.writerows(output_rows)