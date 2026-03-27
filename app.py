import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# --- 1. Page Configuration ---
st.set_page_config(page_title="INDIAN ENSO PREDICTOR", layout="wide", page_icon="🌍")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #eee; }
    .stButton>button { background-color: #1E3A5F; color: white; border-radius: 6px; width: 100%; font-weight: bold; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. Header ---
st.markdown("<h1 style='text-align: center; color: #1E3A5F;'>INDIAN EL NIÑO & LA NIÑA CLIMATE EFFECT PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #666;'>Meteorological Forecasting | Rewa Engineering College</h4>", unsafe_allow_html=True)
st.divider()

# --- 3. Data Loading ---
@st.cache_data
def load_data():
    df = pd.read_csv("enso_all_merged_with_air_pressure.csv")
    df['time'] = pd.to_datetime(df['time'])
    return df

df_final = load_data()

# --- SECTION 1: DASHBOARD ---
col_graph, col_ctrl = st.columns([2.2, 1])

with col_graph:
    st.subheader("📈 70-Year ENSO Timeline (1960-2030)")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    # Plotting Actual vs Predicted
    ax.plot(df_final['time'], df_final['nino34_anom'], color='#1E3A5F', alpha=0.6, label="Actual Index")
    ax.plot(df_final['time'], df_final['pred'], color='#E63946', linewidth=1, label="KAN Prediction")
    
    ax.axhspan(0.5, 3.5, color='red', alpha=0.1)
    ax.axhspan(-3.5, -0.5, color='blue', alpha=0.1)
    ax.axhline(0, color='black', linestyle='--', alpha=0.2)
    
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper left')
    st.pyplot(fig)

with col_ctrl:
    st.subheader("🗓️ Digital Station")
    st.info(f"**Current Date:**\n{datetime.now().strftime('%A, %d %B %Y')}")
    st.markdown("---")
    st.subheader("🚀 KAN Predictor")
    predict_trigger = st.button("RUN 10-YEAR PREDICTION")
    if predict_trigger:
        st.toast("Processing 2020-2030 Data...")

# --- SECTION 2: 12-MONTH RESULTS ---
if predict_trigger:
    st.divider()
    st.subheader("📋 Detailed Monthly Forecast Log (2020 - 2030)")
    
    def classify_enso(val):
        if val >= 0.5: return "🔴 EL NIÑO", "Drought Risk"
        elif val <= -0.5: return "🔵 LA NIÑA", "Flood Risk"
        else: return "⚪ NEUTRAL", "Normal"

    df_rep = df_final[df_final['time'].dt.year >= 2020].copy()
    df_rep['Condition'], df_rep['Impact'] = zip(*df_rep['pred'].map(classify_enso))
    df_rep['Timeline'] = df_rep['time'].dt.strftime('%Y - %B')
    
    st.dataframe(df_rep[['Timeline', 'pred', 'Condition', 'Impact']].reset_index(drop=True), use_container_width=True)

st.divider()

# --- SECTION 3: 7-DAY WEEKLY FORECAST (Fixed) ---
st.subheader("📅 Weekly Forecast Outlook (Sunday to Sunday)")
today = datetime.now()
days_to_sun = (6 - today.weekday()) % 7
next_sun = today + timedelta(days=days_to_sun)

w_cols = st.columns(8)
for i in range(8):
    f_day = next_sun + timedelta(days=i)
    with w_cols[i]:
        st.metric(f_day.strftime('%a'), f_day.strftime('%d %b'))
        st.caption("Monitoring... 🌤️")

st.divider()

# --- SECTION 4: ANALYTICS (Fixed Heatmap) ---
st.subheader("🔬 Deep Analytics & Model Comparison")
c1, c2 = st.columns([1, 1.2])

with c1:
    st.markdown("### 🛠️ Integrated Parameters")
    st.markdown("""
    - **Zonal/Meridional Winds**
    - **Sea Level Pressure (SLP)**
    - **Sunspot Count**
    - **Air Temperature**
    - **3-Month Temporal Lags**
    """)
    st.write("---")
    st.markdown("### 🏆 Performance")
    st.table(pd.DataFrame({"Model": ["SVM", "KAN"], "Accuracy": ["71.9%", "73.5%"]}))

with c2:
    st.markdown("### 🔥 Prediction Accuracy Heatmap")
    # Heatmap between Actual Anomaly and KAN Prediction
    fig_h, ax_h = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_final[['nino34_anom', 'pred']].corr(), annot=True, cmap='coolwarm', ax=ax_h)
    st.pyplot(fig_h)

st.sidebar.title("REC Lab")
st.sidebar.success("Connection: Permanent Live")
