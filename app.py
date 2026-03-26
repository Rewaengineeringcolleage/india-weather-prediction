import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="India Weather AI", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("Physics-Informed AI for Climate Forecasting")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

def run_prediction(df):
    # 1. Clean Column Names
    df.columns = df.columns.str.strip().str.lower()
    
    # 2. Identify Required Columns
    features = ['uwnd', 'vwnd', 'slp'] # Core Walker Circulation features
    target = 'nino34_anom'
    
    if target not in df.columns:
        st.error(f"Error: '{target}' column not found in your file!")
        return None, None

    # Filter available features
    found_features = [f for f in features if f in df.columns]
    if not found_features:
        st.error("Error: No valid features (uwnd, slp, etc.) found in file!")
        return None, None

    # 3. Data Preparation
    X = df[found_features].values
    y = df[target].values.reshape(-1, 1)
    
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_s = scaler_x.fit_transform(X)
    y_s = scaler_y.fit_transform(y)
    
    # 4. Train/Test Split (Important for KAN)
    split = int(len(X_s) * 0.8)
    train_input = torch.tensor(X_s[:split], dtype=torch.float32)
    train_label = torch.tensor(y_s[:split], dtype=torch.float32)
    test_input = torch.tensor(X_s[split:], dtype=torch.float32)
    test_label = torch.tensor(y_s[split:], dtype=torch.float32)
    
    dataset = {
        'train_input': train_input,
        'train_label': train_label,
        'test_input': test_input,
        'test_label': test_label
    }
    
    # 5. KAN Model Configuration
    # Input size depends on number of features found
    model = KAN(width=[len(found_features), 2, 1], grid=3, k=3)
    
    with st.spinner('AI is analyzing Ocean-Atmosphere coupling...'):
        model.fit(dataset, steps=5) # Reduced steps for faster web response
    
    # 6. Future Prediction (Next 6 Months)
    last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
    preds = []
    for i in range(6):
        p = model(last_input)
        val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
        preds.append(val)
        
    return df, preds

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    df, predictions = run_prediction(data)
    
    if predictions:
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("6-Month Forecast Graph")
            forecast_df = pd.DataFrame({
                "Month": [f"Month +{i+1}" for i in range(6)],
                "Anomaly": predictions
            })
            st.line_chart(forecast_df.set_index('Month'))
        
        with col2:
            st.subheader("India Impact Table")
            for i, val in enumerate(predictions):
                # ENSO Logic
                if val > 0.5: status = "🔴 El Niño (Risk of Drought)"
                elif val < -0.5: status = "🔵 La Niña (Good Monsoon)"
                else: status = "🟢 Neutral (Normal Rain)"
                st.write(f"**Month {i+1}:** {val:.2f} — {status}")

st.info("💡 **Science Note:** This model tracks the **Walker Circulation**. Changes in Sea Level Pressure (SLP) and Wind (UWND) indicate shifts in India's monsoon intensity.")
