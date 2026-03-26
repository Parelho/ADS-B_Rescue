import csv
import pyModeS as pms
import geopandas as gpd
from shapely.geometry import Point, LineString
import matplotlib.pyplot as plt
import math
import simplekml

file_path = "output.csv"

rows = []

# Read CSV
with open(file_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

points = []

# Decode positions
for i in range(len(rows) - 1):
    row = rows[i]

    if row["lat"] and row["lon"]:
        points.append(Point(row["lon"], row["lat"]))

    if i > 9000:
        break

# Create GeoDataFrame of points
gdf_points = gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")

# Create a LineString (flight path)
if len(points) > 1:
    line = LineString(points)
    gdf_line = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")

# Plot
fig, ax = plt.subplots(figsize=(10, 10))

# Plot path
if len(points) > 1:
    gdf_line.plot(ax=ax)

# Plot points
gdf_points.plot(ax=ax, markersize=5)

# plt.title("Flight Path")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.show()

kml = simplekml.Kml()
# Add points (optional)
for pt in points:
    kml.newpoint(coords=[(pt.x, pt.y)])

# Add flight path (line)
if len(points) > 1:
    coords = [(pt.x, pt.y) for pt in points]
    kml.newlinestring(name="Flight Path", coords=coords)

# Save file
kml.save("flight_path.kml")