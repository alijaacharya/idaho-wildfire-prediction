import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import json
from streamlit_folium import st_folium
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Idaho Wildfire Risk", layout="wide")
st.title("Idaho Wildfire Risk Dashboard")
st.caption("Summer 2026 seasonal forecast · County-level predictions")

model = joblib.load('data/processed/model_A_janmay.pkl')
features = joblib.load('data/processed/model_A_features.pkl')

master = pd.read_csv('data/processed/master_county_level.csv')
master.columns = pd.Index(master.columns.tolist())
master = master.loc[:, ~master.columns.duplicated()].reset_index(drop=True)

le = LabelEncoder()
master['county_id'] = le.fit_transform(master['COUNTY'])

latest = master.dropna(subset=features)
latest = latest.sort_values('FIRE_YEAR').groupby('COUNTY').last().reset_index()
latest.columns = pd.Index(latest.columns.tolist())
latest = latest.loc[:, ~latest.columns.duplicated()].reset_index(drop=True)
latest = latest.rename(columns={'COUNTY': 'county'})

latest['fire_prob'] = model.predict_proba(latest[features])[:, 1]
latest['risk_score'] = (latest['fire_prob'] * 100).round(1)
latest['risk_level'] = latest['risk_score'].apply(
    lambda x: 'Critical' if x >= 75 else 'High' if x >= 50 else 'Medium' if x >= 25 else 'Low'
)
latest = latest.sort_values('risk_score', ascending=False).reset_index(drop=True)

st.sidebar.header("Filters")
risk_threshold = st.sidebar.slider("Minimum risk score", 0, 100, 0)
filtered = latest[latest['risk_score'] >= risk_threshold].copy().reset_index(drop=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties analyzed", len(latest))
col2.metric("Critical risk (>75)", len(latest[latest['risk_score'] > 75]))
col3.metric("High risk (>50)", len(latest[latest['risk_score'] > 50]))
col4.metric("Avg risk score", f"{latest['risk_score'].mean():.0f}")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Idaho County Risk Map")

    with open('data/raw/idaho_only.geojson', 'r') as f:
        idaho_geo = json.load(f)

    risk_dict = dict(zip(filtered['county'], filtered['risk_score']))

    m = folium.Map(location=[44.5, -114.5], zoom_start=5)

    def get_color(county_name):
        score = risk_dict.get(county_name, 0)
        if score >= 75: return '#E24B4A'
        elif score >= 50: return '#D85A30'
        elif score >= 25: return '#EF9F27'
        else: return '#639922'

    folium.GeoJson(
        idaho_geo,
        style_function=lambda f: {
            'fillColor': get_color(f['properties']['NAME']),
            'color': 'white',
            'weight': 1,
            'fillOpacity': 0.7
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME'],
            aliases=['County:'],
            localize=True
        )
    ).add_to(m)

    st_folium(m, width=700, height=500)

with col_right:
    st.subheader("County Details")
    if len(filtered) == 0:
        st.warning("No counties match this threshold. Lower the slider.")
    else:
        county_list = filtered['county'].drop_duplicates().values.tolist()
        selected = st.selectbox("Select a county", county_list)
        county_data = filtered[filtered['county'] == selected].copy().reset_index(drop=True)
        row = county_data.iloc[0]
        st.metric("Risk Score", f"{row['risk_score']}/100")
        st.metric("Risk Level", row['risk_level'])
        st.metric("Year of data", int(row['FIRE_YEAR']))
        st.metric("Avg Temp Jan", f"{row['avg_temp_month1']:.1f}°F")
        st.metric("Avg Temp March", f"{row['avg_temp_month3']:.1f}°F")
        st.metric("Avg Temp May", f"{row['avg_temp_month5']:.1f}°F")
        st.metric("Precip Anomaly Jan", f"{row['precip_anomaly_month1']:.2f} in")
        st.metric("Snow Anomaly Jan", f"{row['snow_anomaly_month1']:.2f} in")

st.divider()
st.subheader("Full County Table")
display_cols = ['county', 'risk_score', 'risk_level', 'FIRE_YEAR',
                'avg_temp_month1', 'avg_temp_month3', 'precip_anomaly_month3']
st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

st.caption("Model: Random Forest · Jan-May features · Threshold 0.2 · Accuracy: 59.6% (>1000 acres) | 84.9% (>10000 acres)")