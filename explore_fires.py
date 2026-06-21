import geopandas as gpd
import pandas as pd

# Load the NIFC fire perimeters file
fires = gpd.read_file("data/raw/nifc/WFIGS_Interagency_Perimeters.geojson")

# See what columns we have
print("Columns:", fires.columns.tolist())
print("Total records:", len(fires))

# Filter to Idaho only
idaho_fires = fires[fires['STATE'] == 'ID']
print("Idaho records:", len(idaho_fires))

# Save Idaho-only file
idaho_fires.to_file("data/raw/nifc/idaho_fire_perimeters.geojson", driver="GeoJSON")
print("Saved Idaho fires!")