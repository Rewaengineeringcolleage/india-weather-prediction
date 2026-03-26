import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

st.set_page_config(page_title="India Weather AI Dashboard", layout="wide")

st.title("🇮🇳 India ENSO & Monsoon Predictor (1970 - Future)")
st.markdown("### Advanced KAN Model for Climate Forecasting")

@st.cache_data
def load_internal_data():
    # File name must match exactly what is on GitHub
    file_path = "enso_all_merged_data (1) FINALE.csv" 
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    return df[df['time'].dt.year >= 1970]

try:
    df = load_internal_data()
    
    if st.sidebar.button("🚀 Calculate Future Forecast"):
        # --- AI Training Logic ---
        features = ['uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        # KAN Model Setup
        model = KAN(width=[3, 3, 1], grid=5, k=3)
        dataset = {
            'train_input': torch.tensor(X_s[:-24], dtype=torch.float32),
            'train_label': torch.tensor(y_s[:-24], dtype=torch.float32),
            'test_input': torch.tensor(X_s[-24:], dtype=torch.float32),
            'test_label': torch.tensor(y_s[-24:], dtype=torch.float32)
        }
        
        with st.spinner("AI is analyzing 50 years of data..."):
            model.fit(dataset, steps=10) # More steps for better accuracy
            
            # Future 12 Months
            last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
            future_dates = pd.date_range(start=df['time'].max(), periods=13, freq='MS')[1:]
            future_preds = []
            for _ in range(12):
                p = model(last_input)
                val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                future_preds.append(val)
        
        # --- Clean & Detailed Graph Visualization ---
        fig = go.Figure()

        # 1. El Niño Zone (Orange Background)
        fig.add_hrect(y0=0.5, y1=3.0, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="El Niño Zone (Drought Risk)", annotation_position="top left")
        
        # 2. La Niña Zone (Blue Background)
        fig.add_hrect(y0=-3.0, y1=-0.5, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="La Niña Zone (Heavy Rain)", annotation_position="bottom left")

        # 3. Historical Data (Gray Line)
        fig.add_trace(go.Scatter(x=df['time'], y=df['nino34_anom'], name="Past Data", line=dict(color='rgba(100,100,100,0.5)', width=1.5), hovertemplate='Date: %{x}<br>Anomaly: %{y:.2f}'))

        # 4. Future AI Prediction (Thick Red Line)
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="AI Future Forecast", mode='lines+markers', line=dict(color='red', width=4), hovertemplate='<b>Forecast</b><br>Month: %{x}<br>Anomaly: %{y:.2f}'))

        # Threshold Dash Lines
        fig.add_hline(y=0.5, line_dash="dash", line_color="red")
        fig.add_hline(y=-0.5, line_dash="dash", line_color="blue")
        fig.add_hline(y=0, line_color="black", width=1)

        fig.update_layout(
            height=600,
            hovermode="x unified", # Isse click karne ki zarurat nahi, mouse le jate hi info dikhegi
            title="Detailed Climate Trend: 1970 to 2027",
            xaxis=dict(rangeslider=dict(visible=True), type="date"), # Range slider for easy zooming
            yaxis=dict(title="Nino 3.4 Anomaly Index", range=[-2.5, 2.5])
        )

        st.plotly_chart(fig, use_container_width=True)

        # Result Details in a Clean Table
        st.subheader("🗓️ Monthly Forecast Breakdown")
        res_df = pd.DataFrame({"Month": future_dates.strftime('%B %Y'), "Anomaly Value": [round(x, 2) for x in future_preds]})
        
        # Categorization logic for the table
        def categorize(val):
            if val > 0.5: return "🔴 El Niño (Dry/Hot)"
            if val < -0.5: return "🔵 La Niña (Strong Monsoon)"
            return "🟢 Neutral (Normal)"
        
        res_df['Climate Condition'] = res_df['Anomaly Value'].apply(categorize)
        st.table(res_df)

except Exception as e:
    st.error(f"Error: Make sure the CSV file is uploaded correctly. Details: {e}")
