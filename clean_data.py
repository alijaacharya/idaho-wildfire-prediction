import pandas as pd
import numpy as np

print("Loading NOAA data...")
noaa = pd.read_csv('data/raw/noaa/noaa_idaho_2013_2015.csv')

# Keep columns we need
noaa = noaa[['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'DATE',
             'TMAX', 'TMIN', 'PRCP', 'AWND', 'SNOW']]

# Convert date
noaa['DATE'] = pd.to_datetime(noaa['DATE'])
noaa['YEAR'] = noaa['DATE'].dt.year

# Summer months only (June-August)
summer = noaa[noaa['DATE'].dt.month.isin([6, 7, 8])]

# Aggregate with extremes
station_avg = summer.groupby(['STATION', 'NAME', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'YEAR']).agg(
    avg_temp_f=('TMAX', 'mean'),
    max_temp_f=('TMAX', 'max'),
    min_temp_f=('TMIN', 'min'),
    total_precip_in=('PRCP', 'sum'),
    min_precip_day=('PRCP', 'min'),
    avg_wind=('AWND', 'mean'),
    max_wind=('AWND', 'max'),
    total_snow=('SNOW', 'sum')
).reset_index()

print("NOAA cleaned shape:", station_avg.shape)
print("Columns:", station_avg.columns.tolist())
print(station_avg.head(3))

station_avg.to_csv('data/processed/noaa_summer_2013_2015.csv', index=False)
print("Saved!")
# Clean county names
print("\nCleaning county names...")
idaho_counties = [
    'Ada', 'Adams', 'Bannock', 'Bear Lake', 'Benewah', 'Bingham', 'Blaine',
    'Boise', 'Bonner', 'Bonneville', 'Boundary', 'Butte', 'Camas', 'Canyon',
    'Caribou', 'Cassia', 'Clark', 'Clearwater', 'Custer', 'Elmore', 'Franklin',
    'Fremont', 'Gem', 'Gooding', 'Idaho', 'Jefferson', 'Jerome', 'Kootenai',
    'Latah', 'Lemhi', 'Lewis', 'Lincoln', 'Madison', 'Minidoka', 'Nez Perce',
    'Oneida', 'Owyhee', 'Payette', 'Power', 'Shoshone', 'Teton', 'Twin Falls',
    'Valley', 'Washington'
]

fires = pd.read_csv('data/processed/idaho_fire_counts.csv')

# Standardize county names to title case
fires['COUNTY'] = fires['COUNTY'].str.strip().str.title()

# Keep only valid Idaho counties
fires_clean = fires[fires['COUNTY'].isin(idaho_counties)]
print("Clean fire records:", len(fires_clean))
print("Unique counties:", fires_clean['COUNTY'].nunique())

fires_clean.to_csv('data/processed/idaho_fire_counts_clean.csv', index=False)
print("Saved clean fire counts!")
# Combine NOAA and fire data
print("\nCombining datasets...")
noaa = pd.read_csv('data/processed/noaa_summer_2013_2015.csv')
fires = pd.read_csv('data/processed/idaho_fire_counts_clean.csv')

# Average NOAA features across all stations per year
noaa_avg = noaa.groupby('YEAR').agg(
    avg_temp_f=('avg_temp_f', 'mean'),
    max_temp_f=('max_temp_f', 'mean'),
    min_temp_f=('min_temp_f', 'mean'),
    total_precip_in=('total_precip_in', 'mean'),
    avg_wind=('avg_wind', 'mean'),
    max_wind=('max_wind', 'mean'),
    elevation=('ELEVATION', 'mean')
).reset_index()

# Merge fire data with NOAA data on year
master = fires.merge(noaa_avg, left_on='FIRE_YEAR', right_on='YEAR', how='left')

# Create fire label - did this county have a significant fire this year?
master['fire_occurred'] = (master['fire_count'] > 0).astype(int)

print("Master dataset shape:", master.shape)
print("Columns:", master.columns.tolist())
print("Fire rate:", master['fire_occurred'].mean().round(2))
print(master.head())

master.to_csv('data/processed/master_dataset.csv', index=False)
print("Master dataset saved!")