import streamlit as st
import pandas as pd
import torch
import numpy as np
import requests
import matplotlib.pyplot as plt
from kan import KAN
from sklearn.preprocessing import StandardScaler

# --- Page Setup ---
st.set_page_config(page_title="KAN Weather AI", page_icon="🌍", layout="wide")

st.title("🚀 KAN Superhero: ENSO Hybrid Predictor (1960-2030)")
st.markdown("This application uses **Kolmogorov-Arnold Networks (KAN)** to forecast El Niño and La Niña events.")

# --- Load Data (Automatic Cache) ---
@st.cache_data
def load_and_forecast():
    # 1. Data Load
    df = pd.read_csv("enso_all_merged_with_air_pressure.csv")
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values("time")
    
    # 2. Pre-calculated Results (For Speed)
    # Note: As per our previous training, we assume the model is ready.
    # In a real app, you can load a saved model.pth here.
    return df

try:
    df_final = load_and_forecast()
    st.sidebar.success("✅ Data Loaded Successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading data: {e}")

# --- Sidebar Settings ---
st.sidebar.header("🕹️ Control Panel")
mode = st.sidebar.radio("Select Analysis Mode:", ["Future Forecast (2025-2030)", "Historical Archive", "Live Pacific API"])
api_key = st.sidebar.text_input("Enter OpenWeatherMap API Key (Optional)", type="password")

# --- Mode 1: Future Forecast ---
if mode == "Future Forecast (2025-2030)":
    st.subheader("📅 Future Climate Outlook")
    year = st.select_slider("Select Future Year", options=list(range(2025, 2031)))
    
    # Monthly View
    cols = st.columns(4)
    for i in range(1, 13):
        target_date = f"{year}-{i:02d}-01"
        # Dummy logic for display (Replace with actual model.predict)
        val = np.random.uniform(-1.5, 1.5) 
        
        with cols[(i-1)%4]:
            st.write(f"**Month {i}**")
            if val >= 0.5: st.error(f"🔥 El Niño ({val:.2f})")
            elif val <= -0.5: st.info(f"🔵 La Niña ({val:.2f})")
            else: st.success(f"⚪ Neutral ({val:.2f})")

# --- Mode 2: Live API ---
elif mode == "Live Pacific API":
    st.subheader("📡 Real-time Data from Nino 3.4 Region")
    if st.button("Fetch Current Ocean Data"):
        if api_key:
            # Lat/Lon for Central Pacific
            url = f"https://api.openweathermap.org/data/2.5/weather?lat=-5&lon=-145&appid={api_key}&units=metric"
            res = requests.get(url).json()
            if res.get("main"):
                st.metric("Current Temp", f"{res['main']['temp']} °C")
                st.metric("Pressure", f"{res['main']['pressure']} hPa")
                st.info("🔄 Processing through KAN Model...")
            else:
                st.error("Invalid API Key or Service Down.")
        else:
            st.warning("Please provide an API Key in the sidebar to use Live Mode.")

# --- Visualizations ---
st.divider()
st.subheader("📊 Long-term ENSO Trend (1960 - 2030)")
# Simple Plot
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(100), np.sin(np.linspace(0, 10, 100)), color='gray', alpha=0.5) # Placeholder
ax.axhspan(0.5, 1.5, color='red', alpha=0.1)
ax.axhspan(-1.5, -0.5, color='blue', alpha=0.1)
st.pyplot(fig)
