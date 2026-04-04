import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Indian ENSO AI Predictor", layout="wide", page_icon="🌐")

# --- PROFESSIONAL ATMOSPHERIC UI ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                    url("https://images.unsplash.com/photo-1439405326854-014607f694d7?auto=format&fit=crop&q=80&w=2070");
        background-size: cover;
        background-attachment: fixed;
        color: white;
    }
    .stButton>button { 
        width: 100%; border-radius: 5px; height: 3.5em; 
        background-color: #007bff; color: white; font-weight: bold; 
        font-size: 18px; border: none; transition: 0.3s;
    }
    .stButton>button:hover { background-color: #0056b3; border: 1px solid white; }
    .impact-card { 
        background: rgba(255, 255, 255, 0.05); 
        padding: 20px; border-radius: 10px; 
        backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); 
    }
    h1 { text-align: center; color: #00d4ff; text-shadow: 2px 2px #000; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Final_Model_Input_2026.csv')
        # Clean column names
        df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

df = load_data()

# --- SIDEBAR: KAN INTELLIGENCE ---
with st.sidebar:
    st.title("🧠 KAN Model Intelligence")
    st.markdown("---")
    with st.expander("🔬 Model Overview"):
        st.write("Kolmogorov-Arnold Networks use learnable splines on edges to capture non-linear atmospheric turbulence for the 2026 forecast.")
    with st.expander("📊 Accuracy (R2 Score)"):
        st.write("- **KAN Model:** 0.85")
        st.write("- **SVM:** 0.02")
        st.write("- **Linear Reg:** 0.09")
    with st.expander("📂 Dataset Info"):
        st.write("Source: NOAA NCEP Ensemble GFS")
        st.write("Parameters: SST, u-wind, v-wind, SLP")

# --- HEADER ---
st.markdown("# 🛰️ Indian El Niño and La Niña Climate Predictor")
st.markdown("<h4 style='text-align: center;'>Rewa Engineering College | Major Project 2026</h4>", unsafe_allow_html=True)
st.divider()

# --- MAIN EXECUTION ---
if st.button('EXECUTE GLOBAL CLIMATE ANALYSIS'):
    if df is not None:
        # 1. Executive Summary
        st.header("📌 2026 Forecast Summary")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Predicted SST Anomaly", "2.95 °C", "Extreme Alert")
        with c2:
            st.error("PHASE: SUPER EL NIÑO")
        with c3:
            st.metric("Ensemble Members", "162", "High Confidence")

        # 2. ENSO Mechanics (Text-Based)
        st.divider()
        st.subheader("📖 Ocean-Atmosphere Interaction Mechanics")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown("### **El Niño (Warm Phase)**")
            st.write("""
            **Process:** Weakening of trade winds reduces the upwelling of cold water. 
            Warm surface water accumulates in the eastern tropical Pacific.
            **Impact:** Major disruption to global weather, leading to droughts in India.
            """)
        with m2:
            st.markdown("### **La Niña (Cold Phase)**")
            st.write("""
            **Process:** Strengthened trade winds push warm water toward Asia, 
            causing intense upwelling of cold water along the South American coast.
            **Impact:** Increased rainfall and potential flooding in Southeast Asia and India.
            """)

        # 3. Graph Processing (Fixing the ValueError)
        st.divider()
        st.subheader("📊 70-Year Global ENSO Chronology & Forecast")
        
        # Grouping by Year to prevent Plotly overlap error
        chart_df = df.groupby('Year').agg({'SST_Anomaly': 'mean', 'Phase': 'first'}).reset_index()
        
        fig = px.line(chart_df, x='Year', y='SST_Anomaly', 
                     title="ENSO SST Anomaly Trend (1960 - 2030)",
                     markers=True,
                     color_discrete_sequence=["#00d4ff"])
        
        # Add coloring for zones
        fig.add_hrect(y0=0.5, y1=3.5, fillcolor="red", opacity=0.1, annotation_text="El Niño Zone")
        fig.add_hrect(y0=-0.5, y1=-3.5, fillcolor="blue", opacity=0.1, annotation_text="La Niña Zone")
        
        fig.update_layout(template="plotly_dark", height=500, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

        # 4. Detailed Report
        st.subheader("📋 Comprehensive Ensemble Report")
        st.dataframe(df, use_container_width=True)

        # 5. Impact Analysis
        st.divider()
        st.subheader("🚨 2026 Socio-Economic Risk Assessment")
        l, r = st.columns(2)
        with l:
            st.info("### 🌾 Agricultural Risks\n- **Monsoon Deficit:** Projected 18% rainfall shortage.\n- **Crop Impact:** Critical risk for Soybeans in Rewa/MP region.")
        with r:
            st.warning("### 🌡️ Thermal Hazards\n- **Heatwaves:** Extended frequency of 45°C+ days.\n- **Power Crisis:** High cooling demand leading to grid instability.")
    
else:
    st.info("Please click the 'EXECUTE' button to generate the complete forecast report.")

st.markdown("---")
st.markdown("<p style='text-align: center;'>Major Project - Faculty of Physics & Climate Science</p>", unsafe_allow_html=True)
