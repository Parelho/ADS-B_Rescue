import csv

input_file = "output_vectors.csv"
output_file = "trajectory.csv"

flights = {}

# =========================
# LOAD CSV
# =========================
with open(input_file, newline='') as f:

    # FIX:
    # Use DictReader because CSV now has headers
    reader = csv.DictReader(f)

    for row in reader:
        try:
            time = int(row["time"])

            icao24 = row["icao24"].strip()

            callsign = row["callsign"].strip()

            lat = row["lat"]
            lon = row["lon"]

            velocity = row["velocity"]
            heading = row["heading"]
            vertrate = row["vertrate"]

            onground = (
                row["onground"]
                .strip()
                .lower()
            )

            baroaltitude = row["baroaltitude"]
            geoaltitude = row["geoaltitude"]

            # =========================
            # SKIP INVALID ROWS
            # =========================
            if (
                lat == "" or
                lon == "" or
                velocity == "" or
                heading == "" or
                vertrate == ""
            ):
                continue

            # Skip grounded aircraft
            if onground == "true":
                continue

            # =========================
            # GROUP FLIGHTS
            # =========================
            key = (
                icao24,
                callsign
            )

            if key not in flights:
                flights[key] = []

            flights[key].append({
                "time": time,
                "icao24": icao24,
                "callsign": callsign,
                "lat": float(lat),
                "lon": float(lon),
                "velocity": float(velocity),
                "heading": float(heading),
                "vertrate": float(vertrate),
                "baroaltitude":
                    float(baroaltitude)
                    if baroaltitude != ""
                    else 0.0,
                "geoaltitude":
                    float(geoaltitude)
                    if geoaltitude != ""
                    else 0.0
            })

        except Exception:
            continue

# =========================
# SORT FLIGHTS
# =========================
for key in flights:

    flights[key] = sorted(
        flights[key],
        key=lambda x: x["time"]
    )

# =========================
# EXPORT
# =========================
output_rows = []

for key, flight in flights.items():

    # Ignore tiny trajectories
    if len(flight) < 30:
        continue

    for row in flight:

        output_rows.append(row)

output_rows = sorted(
    output_rows,
    key=lambda x: (
        x["icao24"],
        x["time"]
    )
)

# =========================
# SAVE CSV
# =========================
with open(output_file, "w", newline='') as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "time",
            "icao24",
            "callsign",
            "lat",
            "lon",
            "velocity",
            "heading",
            "vertrate",
            "baroaltitude",
            "geoaltitude"
        ]
    )

    writer.writeheader()

    writer.writerows(output_rows)

print(f"Saved {output_file}")

print(f"Flights: {len(flights)}")
print(f"Rows: {len(output_rows)}")