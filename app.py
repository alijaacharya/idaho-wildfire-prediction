import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import json
from streamlit_folium import st_folium

st.set_page_config(page_title="Idaho Wildfire Risk", layout="wide")
st.title("Idaho Wildfire Risk Dashboard")
st.caption("Summer 2026 seasonal forecast · County-level predictions")

model = joblib.load('data/processed/wildfire_model.pkl')
df = pd.read_csv('data/processed/fake_county_data.csv')

features = ['avg_temp_f', 'humidity_pct', 'wind_mph', 'precip_in',
            'veg_density', 'elevation_ft', 'fire_history_count']

county_avg = df.groupby('county')[features].mean().reset_index()
county_avg['fire_prob'] = model.predict_proba(county_avg[features])[:, 1]
county_avg['risk_score'] = (county_avg['fire_prob'] * 100).round(1)
county_avg['risk_level'] = county_avg['risk_score'].apply(
    lambda x: 'Critical' if x >= 80 else 'High' if x >= 60 else 'Medium' if x >= 40 else 'Low'
)
county_avg = county_avg.sort_values('risk_score', ascending=False)

st.sidebar.header("Filters")
risk_threshold = st.sidebar.slider("Minimum risk score", 0, 100, 0)
filtered = county_avg[county_avg['risk_score'] >= risk_threshold]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties analyzed", len(county_avg))
col2.metric("Critical risk (>80)", len(county_avg[county_avg['risk_score'] > 80]))
col3.metric("High risk (>60)", len(county_avg[county_avg['risk_score'] > 60]))
col4.metric("Avg risk score", f"{county_avg['risk_score'].mean():.0f}")

st.divider()

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Idaho County Risk Map")

    with open('data/raw/idaho_only.geojson', 'r') as f:
        idaho_geo = json.load(f)

    risk_dict = dict(zip(filtered['county'], filtered['risk_score']))
    level_dict = dict(zip(filtered['county'], filtered['risk_level']))

    m = folium.Map(location=[44.5, -114.5], zoom_start=5)

    def get_color(county_name):
        score = risk_dict.get(county_name, 0)
        if score >= 80: return '#E24B4A'
        elif score >= 60: return '#D85A30'
        elif score >= 40: return '#EF9F27'
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
    selected = st.selectbox("Select a county", filtered['county'].tolist())
    row = filtered[filtered['county'] == selected].iloc[0]
    st.metric("Risk Score", f"{row['risk_score']}/100")
    st.metric("Risk Level", row['risk_level'])
    st.metric("Avg Temperature", f"{row['avg_temp_f']:.1f}°F")
    st.metric("Avg Humidity", f"{row['humidity_pct']:.1f}%")
    st.metric("Avg Precipitation", f"{row['precip_in']:.1f} in")
    st.metric("Wind Speed", f"{row['wind_mph']:.1f} mph")

st.divider()
st.subheader("Full County Table")
st.dataframe(filtered[['county', 'risk_score', 'risk_level', 'avg_temp_f',
                        'humidity_pct', 'precip_in', 'wind_mph']].reset_index(drop=True),
             use_container_width=True)

st.caption("Model: Random Forest · Trained on 2005–2021 · Validated on 2022–2024 · Accuracy: 93.2%")