import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="India Weather AI 1970-Future", layout="wide")

# Title
st.title("🇮🇳 India ENSO & Monsoon Predictor (1970 - Future)")

# Function to load data directly from GitHub/Folder
@st.cache_data
def load_internal_data():
    # File ka naam wahi rakhein jo GitHub par hai
    file_path = "enso_all_merged_data (1) FINALE.csv" 
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    # Filter for data from 1970 onwards
    df = df[df['time'].dt.year >= 1970]
    return df

try:
    df = load_internal_data()
    st.success(f"Loaded Data from {df['time'].dt.year.min()} to {df['time'].dt.year.max()}")
    
    if st.button("🚀 Run AI Prediction"):
        # Data Prep
        features = ['uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        # Split for KAN
        split = int(len(X_s) * 0.8)
        dataset = {
            'train_input': torch.tensor(X_s[:split], dtype=torch.float32),
            'train_label': torch.tensor(y_s[:split], dtype=torch.float32),
            'test_input': torch.tensor(X_s[split:], dtype=torch.float32),
            'test_label': torch.tensor(y_s[split:], dtype=torch.float32)
        }
        
        # KAN Model
        model = KAN(width=[3, 2, 1], grid=3, k=3)
        with st.spinner("AI is analyzing decades of climate patterns..."):
            model.fit(dataset, steps=5)
            
            # Future Forecast (Next 12 Months)
            last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
            future_dates = pd.date_range(start=df['time'].max(), periods=13, freq='M')[1:]
            future_preds = []
            for _ in range(12):
                p = model(last_input)
                val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                future_preds.append(val)
        
        # --- Visualization with Plotly ---
        fig = go.Figure()
        
        # Historical Data
        fig.add_trace(go.Scatter(x=df['time'], y=df['nino34_anom'], name="Historical Data", line=dict(color='gray')))
        
        # Future Forecast
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="AI Future Forecast", line=dict(color='red', width=4)))
        
        # Threshold Lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange", annotation_text="El Niño Threshold")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="blue", annotation_text="La Niña Threshold")
        
        fig.update_layout(title="ENSO Trend: 1970 to Future", xaxis_title="Year", yaxis_title="Nino 3.4 Anomaly")
        st.plotly_chart(fig, use_container_width=True)
        
        # Result Table
        st.subheader("Future 12-Month Outlook")
        res_df = pd.DataFrame({"Month": future_dates.strftime('%B %Y'), "Predicted Anomaly": future_preds})
        st.dataframe(res_df.style.highlight_max(axis=0))

except Exception as e:
    st.error(f"Error loading file: Ensure the CSV is in your GitHub folder. Error: {e}")
