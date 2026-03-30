import csv
import pyModeS as pms

file_path = "output.csv"

rows = []

# Read CSV
with open(file_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["hour"] == "1774461600":
            rows.append(row)

for i in range(len(rows) - 1):
    row0 = rows[i]
    row1 = rows[i + 1]

    if row0["icao24"] != row1["icao24"]:
        continue

    if row0["odd"] == row1["odd"]:
        continue

    msg0 = row0["rawmsg"]
    msg1 = row1["rawmsg"]

    t0 = float(row0["mintime"])
    t1 = float(row1["mintime"])

    try:
        lat, lon = pms.adsb.position(msg0, msg1, t0, t1)
        print(f"lat={lat}, lon={lon}")
    except:
        pass