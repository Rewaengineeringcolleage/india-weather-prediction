import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="India Weather AI", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")

uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

def run_prediction(df):
    # Column names ko clean karna (Spaces hatana aur small letters)
    df.columns = df.columns.str.strip().str.lower()
    
    # Check if required columns exist
    required = ['uwnd', 'vwnd', 'slp', 'nino34_anom']
    found_cols = [c for c in required if c in df.columns]
    
    if len(found_cols) < 3:
        st.error(f"Missing Columns! Required: {required}. Found: {df.columns.tolist()}")
        return None, None

    # Time conversion
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    X = df[found_cols[:-1]].values
    y = df['nino34_anom'].values.reshape(-1, 1)
    
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_s = scaler_x.fit_transform(X)
    y_s = scaler_y.fit_transform(y)
    
    # KAN Model
    model = KAN(width=[len(found_cols)-1, 2, 1], grid=3, k=3)
    dataset = {'train_input': torch.tensor(X_s, dtype=torch.float32), 
               'train_label': torch.tensor(y_s, dtype=torch.float32)}
    
    with st.spinner('AI is analyzing Pacific Ocean patterns...'):
        model.fit(dataset, steps=5)
    
    # Next 6 Months Prediction
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
        st.subheader("6-Month ENSO Forecast")
        forecast_df = pd.DataFrame({"Month": [f"Month {i+1}" for i in range(6)], "Anomaly": predictions})
        st.line_chart(forecast_df.set_index('Month'))
        st.table(forecast_df)
