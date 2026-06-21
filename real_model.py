import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load fire data
df = pd.read_csv('data/processed/idaho_fire_counts_clean.csv')

# Features from fire history only
df['fire_occurred'] = (df['fire_count'] > 0).astype(int)
df['large_fire'] = (df['total_acres'] > 1000).astype(int)
df['log_acres'] = np.log1p(df['total_acres'])

# Use previous year fire history as features
df = df.sort_values(['COUNTY', 'FIRE_YEAR'])
df['prev_fire_count'] = df.groupby('COUNTY')['fire_count'].shift(1)
df['prev_acres'] = df.groupby('COUNTY')['total_acres'].shift(1)
df = df.dropna()

features = ['FIRE_YEAR', 'prev_fire_count', 'prev_acres', 'log_acres']
X = df[features]
y = df['fire_occurred']

# Train on 2005-2012, test on 2013-2015
train = df[df['FIRE_YEAR'] <= 2012]
test = df[df['FIRE_YEAR'] > 2012]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train[features], train['fire_occurred'])

y_pred = model.predict(test[features])
print("Real model accuracy:", accuracy_score(test['fire_occurred'], y_pred))
print(classification_report(test['fire_occurred'], y_pred))