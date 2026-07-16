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

# Load 2026 predictions
predictions = pd.read_csv('data/processed/predictions_2026.csv')
predictions = predictions.rename(columns={'COUNTY': 'county'})
predictions['risk_level'] = predictions['risk_score'].apply(
    lambda x: 'Critical' if x >= 75 else 'High' if x >= 50 else 'Medium' if x >= 25 else 'Low'
)
predictions = predictions.sort_values('risk_score', ascending=False).reset_index(drop=True)

st.sidebar.header("Filters")
risk_threshold = st.sidebar.slider("Minimum risk score", 0, 100, 0)
filtered = predictions[predictions['risk_score'] >= risk_threshold].copy().reset_index(drop=True)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties analyzed", len(predictions))
col2.metric("Critical risk (>75)", len(predictions[predictions['risk_score'] > 75]))
col3.metric("High risk (>50)", len(predictions[predictions['risk_score'] > 50]))
col4.metric("Avg risk score", f"{predictions['risk_score'].mean():.0f}")

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
        county_list = filtered['county'].values.tolist()
        selected = st.selectbox("Select a county", county_list)
        row = filtered[filtered['county'] == selected].iloc[0]
        st.metric("Risk Score", f"{row['risk_score']}/100")
        st.metric("Risk Level", row['risk_level'])
        st.metric("Year", "2026")
        st.metric("Avg Temp Jan", f"{row['avg_temp_month1']:.1f}°F")
        st.metric("Avg Temp March", f"{row['avg_temp_month3']:.1f}°F")
        st.metric("Avg Temp May", f"{row['avg_temp_month5']:.1f}°F")
        st.metric("Precip Anomaly Jan", f"{row['precip_anomaly_month1']:.2f} in")
        st.metric("Snow Anomaly Jan", f"{row['snow_anomaly_month1']:.2f} in")

st.divider()
st.subheader("Full County Table — 2026 Predictions")
display_cols = ['county', 'risk_score', 'risk_level',
                'avg_temp_month1', 'avg_temp_month3', 'precip_anomaly_month3']
st.dataframe(filtered[display_cols].reset_index(drop=True), use_container_width=True)

st.caption("Model: Random Forest · Jan-May 2026 data · Accuracy: 59.6% (>1000 acres) | 84.9% (>10000 acres)")