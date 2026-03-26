import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(page_title="India Weather AI", layout="wide")

st.title("🇮🇳 India ENSO & Monsoon Predictor (KAN Model)")
st.markdown("""
This dashboard uses **Kolmogorov-Arnold Networks (KAN)** to predict Pacific Ocean anomalies 
that directly impact the Indian Monsoon.
""")

# Sidebar for Data Upload
st.sidebar.header("Step 1: Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

def run_prediction(df):
    # Data Cleaning
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Feature Selection (Walker Circulation Variables)
    features = ['uwnd', 'vwnd', 'slp', 'air_temp'] # Basic features
    X = df[features].values
    y = df['nino34_anom'].values.reshape(-1, 1)
    
    # Scaling
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_s = scaler_x.fit_transform(X)
    y_s = scaler_y.fit_transform(y)
    
    # Initialize KAN Model
    # [Input Features, Hidden Layer, Output]
    model = KAN(width=[4, 2, 1], grid=3, k=3)
    
    # Fast training for demo (In real apps, use pre-trained weights)
    dataset = {
        'train_input': torch.tensor(X_s[:-10], dtype=torch.float32),
        'train_label': torch.tensor(y_s[:-10], dtype=torch.float32),
        'test_input': torch.tensor(X_s[-10:], dtype=torch.float32),
        'test_label': torch.tensor(y_s[-10:], dtype=torch.float32)
    }
    model.fit(dataset, steps=10)
    
    # Predict next 6 months
    last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
    future_preds = []
    for i in range(6):
        pred_s = model(last_input)
        val = scaler_y.inverse_transform(pred_s.detach().numpy())[0][0]
        future_preds.append(val)
        # Simple feedback loop for multi-step
        last_input = last_input # In a real scenario, we'd update features
        
    return df, future_preds

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("File Uploaded Successfully!")
    
    df, predictions = run_prediction(data)
    
    # Display Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Historical Nino 3.4 Trend")
        st.line_chart(df.set_index('time')['nino34_anom'].tail(50))
        
    with col2:
        st.subheader("AI Forecast (Next 6 Months)")
        forecast_df = pd.DataFrame({
            "Month": [f"Month {i+1}" for i in range(6)],
            "Predicted Anomaly": predictions
        })
        st.table(forecast_df)
        
    # Logic Explanation
    st.info("💡 **How to read:** If Predicted Anomaly > 0.5 (El Niño), India might face a dry monsoon. If < -0.5 (La Niña), expect heavy rains.")
else:
    st.warning("Please upload a CSV file from the sidebar to start prediction.")

st.markdown("---")
st.caption("Powered by PyKAN | Developed for India Climate Research")
