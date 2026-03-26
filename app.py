import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Page Setup
st.set_page_config(page_title="India Weather AI", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("### 1970 to 2027 Climate Outlook (Historical & AI Forecast)")

# Internal Data Loading
@st.cache_data
def load_data():
    # Make sure this filename matches exactly on your GitHub
    file_path = "enso_all_merged_data (1) FINALE.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    # Filter for data from 1970 onwards as requested
    return df[df['time'].dt.year >= 1970]

try:
    df = load_data()
    
    # Simple UI: Just one button to generate everything
    if st.button("📊 Show Climate Prediction"):
        
        # --- Background AI Processing (Silent Mode) ---
        features = ['uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        # KAN Model Configuration
        model = KAN(width=[3, 3, 1], grid=5, k=3)
        dataset = {
            'train_input': torch.tensor(X_s, dtype=torch.float32),
            'train_label': torch.tensor(y_s, dtype=torch.float32),
            'test_input': torch.tensor(X_s[-5:], dtype=torch.float32),
            'test_label': torch.tensor(y_s[-5:], dtype=torch.float32)
        }
        
        with st.spinner("Analyzing Global Climate Patterns..."):
            # Silent training: no logs on screen
            model.fit(dataset, steps=10, display_metrics=False) 
            
            # Forecast next 12 months
            last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
            future_dates = pd.date_range(start=df['time'].max(), periods=13, freq='MS')[1:]
            future_preds = []
            for _ in range(12):
                p = model(last_input)
                val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                future_preds.append(val)

        # --- High-Quality Interactive Graph ---
        fig = go.Figure()
        
        # Zone Highlighting
        fig.add_hrect(y0=0.5, y1=2.5, fillcolor="red", opacity=0.08, line_width=0, annotation_text="El Niño (Hot/Dry)")
        fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor="blue", opacity=0.08, line_width=0, annotation_text="La Niña (Strong Monsoon)")
        
        # Past Historical Line
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['nino34_anom'], 
            name="Historical Trend", 
            line=dict(color='rgba(150,150,150,0.6)', width=1.5),
            hovertemplate='Date: %{x}<br>Index: %{y:.2f}'
        ))
        
        # Future AI Line (High Visibility)
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_preds, 
            name="AI Future Forecast", 
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(size=8),
            hovertemplate='<b>Forecast</b><br>Month: %{x}<br>Index: %{y:.2f}'
        ))

        # Horizontal Center Line (Fixed the bug here: line_width=1)
        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            height=600,
            hovermode="x unified",
            xaxis=dict(title="Year", rangeslider=dict(visible=True)),
            yaxis=dict(title="Nino 3.4 Anomaly Index", range=[-2.5, 2.5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Result Summary ---
        st.subheader("🗓️ Next 12 Months Outlook")
        res_df = pd.DataFrame({
            "Month": future_dates.strftime('%B %Y'), 
            "Index Value": [round(float(x), 2) for x in future_preds]
        })
        
        def get_status(val):
            if val > 0.5: return "🔴 El Niño Risk (Dry Conditions)"
            if val < -0.5: return "🔵 La Niña Expected (Good Rain)"
            return "🟢 Neutral (Normal Rain)"
        
        res_df['Impact on India'] = res_df['Index Value'].apply(get_status)
        st.table(res_df)

except Exception as e:
    st.error(f"Setup Error: Please ensure 'enso_all_merged_data (1) FINALE.csv' is in your GitHub. Error Details: {e}")
