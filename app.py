import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="India Weather AI", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("### 1970 to 2027 Climate Outlook (Historical & AI Forecast)")

# 2. Loading Data from GitHub
@st.cache_data
def load_data():
    # Make sure this filename is EXACTLY the same on GitHub
    file_path = "enso_all_merged_data (1) FINALE.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    return df[df['time'].dt.year >= 1970]

try:
    df = load_data()
    
    # Simple UI: One Button
    if st.button("📊 Show Climate Prediction"):
        
        # --- Background Data Prep ---
        features = ['uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        # Prepare Dataset for KAN
        # We use a small test set to satisfy the model's requirement
        train_input = torch.tensor(X_s[:-12], dtype=torch.float32)
        train_label = torch.tensor(y_s[:-12], dtype=torch.float32)
        test_input = torch.tensor(X_s[-12:], dtype=torch.float32)
        test_label = torch.tensor(y_s[-12:], dtype=torch.float32)
        
        dataset = {
            'train_input': train_input,
            'train_label': train_label,
            'test_input': test_input,
            'test_label': test_label
        }
        
        with st.spinner("AI is calculating trends (this may take 10-20 seconds)..."):
            # Initialize KAN Model
            model = KAN(width=[3, 3, 1], grid=3, k=3)
            
            # Training without the 'display_metrics' flag to avoid 'bool' error
            model.fit(dataset, steps=5) 
            
            # Forecast next 12 months
            last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
            future_dates = pd.date_range(start=df['time'].max(), periods=13, freq='MS')[1:]
            future_preds = []
            
            # Recursive prediction loop
            for _ in range(12):
                with torch.no_grad():
                    p = model(last_input)
                    val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                    future_preds.append(val)
                    # We reuse the last input for simplicity in this demo
                    last_input = last_input 

        # --- Professional Visualization ---
        fig = go.Figure()
        
        # Colored Background Zones
        fig.add_hrect(y0=0.5, y1=2.5, fillcolor="red", opacity=0.08, line_width=0, annotation_text="El Niño (Dry)")
        fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor="blue", opacity=0.08, line_width=0, annotation_text="La Niña (Wet)")
        
        # Historical Line
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['nino34_anom'], 
            name="Past Trend", 
            line=dict(color='gray', width=1.5)
        ))
        
        # Future AI Forecast Line
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_preds, 
            name="AI Future Forecast", 
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(size=8)
        ))

        # Horizontal Zero Line
        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            height=600,
            hovermode="x unified",
            xaxis=dict(title="Year", rangeslider=dict(visible=True)),
            yaxis=dict(title="Nino 3.4 Anomaly Index", range=[-2.5, 2.5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Final Outcome Table ---
        st.subheader("🗓️ Forecast Summary")
        res_df = pd.DataFrame({
            "Month": future_dates.strftime('%B %Y'), 
            "Index": [round(float(x), 2) for x in future_preds]
        })
        
        def check_impact(v):
            if v > 0.5: return "🔴 El Niño (Possible Drought)"
            if v < -0.5: return "🔵 La Niña (Good Rain)"
            return "🟢 Neutral (Normal Rain)"
            
        res_df['India Impact'] = res_df['Index'].apply(check_impact)
        st.table(res_df)

except Exception as e:
    st.error(f"Error loading dashboard: {e}")
    st.info("Check if 'enso_all_merged_data (1) FINALE.csv' exists in your GitHub repository.")
