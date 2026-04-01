import csv
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import simplekml
import re

file_path = "output.csv"

rows = []

with open(file_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["day"] and int(row["day"]) >= 1485734400:
            rows.append(row)

points = []

# Function to parse track field
def parse_track(track_str):
    coords = []
    
    matches = re.findall(
        r'time=([0-9]+), latitude=([\-0-9.]+), longitude=([\-0-9.]+)',
        track_str
    )
    
    for t, lat, lon in matches:
        coords.append((int(t), float(lat), float(lon)))
    
    return coords

output_rows = []
for row in rows:
    track = row["track"]
    
    if track:
        coords = parse_track(track)
        
        for t, lat, lon in coords:
            output_rows.append({
                "time": t,
                "lat": lat,
                "lon": lon
            })
            points.append(Point(lon, lat))

# Create GeoDataFrame of points
gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

# Create LineString
if len(points) > 1:
    line = LineString(points)
    gdf_line = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

if len(points) > 1:
    gdf_line.plot(ax=ax)

gdf_points.plot(ax=ax, markersize=5)

# plt.show()

# Export to KML
kml = simplekml.Kml()

for pt in points:
    kml.newpoint(coords=[(pt.x, pt.y)])

if len(points) > 1:
    coords = [(pt.x, pt.y) for pt in points]
    kml.newlinestring(name="Flight Path", coords=coords)

kml.save("flight_path.kml")

output_rows = sorted(output_rows, key=lambda x: x["time"])
with open("trajectory.csv", "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["time", "lat", "lon"])
    writer.writeheader()
    writer.writerows(output_rows)