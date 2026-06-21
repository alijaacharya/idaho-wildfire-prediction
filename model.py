import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Fake data - 44 Idaho counties, multiple years
np.random.seed(42)

counties = [

    'Ada', 'Adams', 'Bannock', 'Bear Lake', 'Benewah', 'Bingham', 'Blaine',
    'Boise', 'Bonner', 'Bonneville', 'Boundary', 'Butte', 'Camas', 'Canyon',
    'Caribou', 'Cassia', 'Clark', 'Clearwater', 'Custer', 'Elmore', 'Franklin',
    'Fremont', 'Gem', 'Gooding', 'Idaho', 'Jefferson', 'Jerome', 'Kootenai',
    'Latah', 'Lemhi', 'Lewis', 'Lincoln', 'Madison', 'Minidoka', 'Nez Perce',
    'Oneida', 'Owyhee', 'Payette', 'Power', 'Shoshone', 'Teton', 'Twin Falls',
    'Valley', 'Washington'
]

rows = []
for year in range(2005, 2025):
    for county in counties:
        avg_temp     = np.random.uniform(75, 100)
        humidity     = np.random.uniform(10, 40)
        wind_speed   = np.random.uniform(5, 20)
        precip       = np.random.uniform(1, 15)
        veg_density  = np.random.uniform(1, 5)
        elevation    = np.random.uniform(2000, 9000)
        fire_history = np.random.randint(0, 10)

        # Risk formula - higher temp, lower humidity = more fires
        risk = (avg_temp * 0.4 + (40 - humidity) * 0.3 +
                wind_speed * 0.2 - precip * 0.5 + fire_history * 0.3)
        fire_occurred = 1 if risk > 38 else 0

        rows.append({
            'county': county, 'year': year,
            'avg_temp_f': round(avg_temp, 1),
            'humidity_pct': round(humidity, 1),
            'wind_mph': round(wind_speed, 1),
            'precip_in': round(precip, 1),
            'veg_density': round(veg_density, 1),
            'elevation_ft': round(elevation),
            'fire_history_count': fire_history,
            'fire_occurred': fire_occurred
        })

df = pd.DataFrame(rows)
df.to_csv('data/processed/fake_county_data.csv', index=False)
print(f"Dataset: {len(df)} rows, {df['fire_occurred'].mean()*100:.1f}% fire years")

# Features and label
features = ['avg_temp_f', 'humidity_pct', 'wind_mph', 'precip_in',
            'veg_density', 'elevation_ft', 'fire_history_count']
X = df[features]
y = df['fire_occurred']

# Train/test split - train on 2005-2021, test on 2022-2024
train = df[df['year'] <= 2021]
test  = df[df['year'] > 2021]
X_train, y_train = train[features], train['fire_occurred']
X_test,  y_test  = test[features],  test['fire_occurred']

# Train random forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel accuracy: {accuracy*100:.1f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Fire', 'Fire']))

# Feature importance
print("\nFeature Importance:")
for feat, imp in sorted(zip(features, model.feature_importances_), 
                         key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp*100:.1f}%")
# Save the model
import joblib
joblib.dump(model, 'data/processed/wildfire_model.pkl')
print("\nModel saved to data/processed/wildfire_model.pkl")
