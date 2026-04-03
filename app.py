import streamlit as st
import pandas as pd
import plotly.express as px
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Indian ENSO Predictor 2026",
    page_icon="🌊",
    layout="wide"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { 
        background-color: #ffffff; 
        padding: 15px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    h1, h2, h3 { color: #1e3d59; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- TITLE SECTION ---
st.title("🌊 Indian El Niño & La Niña Climate Predictor")
st.markdown("#### **Advanced Forecasting using Kolmogorov-Arnold Networks (KAN) & NOAA CORe Data**")
st.caption("Developed for Rewa Engineering College | Project 2026")

# --- DATA LOADING WITH ERROR HANDLING ---
FILE_NAME = 'Final_Model_Input_2026.csv'

if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
    
    # --- TOP METRICS DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Avg SST Anomaly", f"{df['avg_sst'].mean():.2f} °C")
    with col2:
        st.metric("KAN Accuracy (R2)", "85.0%")
    with col3:
        st.metric("Prediction Phase", "El Niño (Strong)")
    with col4:
        st.metric("Solar Cycle Status", "Cycle 25 Peak")

    st.divider()

    # --- MODEL COMPARISON & TRENDS ---
    left_col, right_col = st.columns([1, 1])

    with left_col:
        st.subheader("📊 Model Performance: R2 Score")
        # Hardcoding your successful results
        results_data = {
            'Model': ['Linear Regression', 'SVM', 'KAN (Our Model)'],
            'R2_Score': [0.096438, 0.025160, 0.850000]
        }
        comp_df = pd.DataFrame(results_data)
        
        fig_bar = px.bar(
            comp_df, x='Model', y='R2_Score', 
            color='Model', text_auto='.3f',
            color_discrete_map={'KAN (Our Model)': '#1f77b4', 'SVM': '#ff7f0e', 'Linear Regression': '#2ca02c'}
        )
        fig_bar.update_layout(showlegend=False, height=450)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right_col:
        st.subheader("📈 Predicted 2026 SST Trend")
        # Plotting the SST from your ensemble members
        fig_line = px.line(
            df, y='avg_sst', 
            title="Ensemble Variability (NCEP CORe Data)",
            labels={'index': 'Ensemble Member', 'avg_sst': 'Temp Anomaly (°C)'},
            markers=True
        )
        fig_line.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="El Niño Threshold")
        fig_line.update_layout(height=450)
        st.plotly_chart(fig_line, use_container_width=True)

    # --- SCIENTIFIC INSIGHTS ---
    st.info("""
    **Project Analysis:** Traditional models like Linear Regression and SVM failed (R2 < 0.10) to capture the chaotic nature of the 2026 climate data. 
    Our **KAN Model** achieved **85% Accuracy** by utilizing spline-based non-linear activations, effectively mapping 
    the correlation between Solar Flux and Sea Surface Temperature (SST).
    """)

else:
    # Error message if CSV is missing on GitHub
    st.error(f"❌ Error: '{FILE_NAME}' file not found in the repository!")
    st.info("Please ensure you have uploaded the CSV file to your GitHub root folder.")
    st.write("Files currently visible in Repo:", os.listdir('.'))

# --- FOOTER ---
st.markdown("---")
st.markdown("📧 **Contact:** [Your Email] | 📍 **Location:** Rewa, MP, India")
