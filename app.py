import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="India Weather Forecast", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("### 1970 to 2027 Climate Outlook (Historical & Future Forecast)")

# 2. Loading Data
@st.cache_data
def load_data():
    file_path = "enso_all_merged_data (1) FINALE.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    return df[df['time'].dt.year >= 1970]

try:
    df = load_data()
    
    if st.button("📊 Generate Long-Term Forecast"):
        
        # --- Processing ---
        features = ['uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        # Internal Model Setup (Silent)
        dataset = {
            'train_input': torch.tensor(X_s[:-12], dtype=torch.float32),
            'train_label': torch.tensor(y_s[:-12], dtype=torch.float32),
            'test_input': torch.tensor(X_s[-12:], dtype=torch.float32),
            'test_label': torch.tensor(y_s[-12:], dtype=torch.float32)
        }
        
        with st.spinner("Processing long-term climate cycles..."):
            model = KAN(width=[3, 3, 1], grid=3, k=3)
            model.fit(dataset, steps=5) 
            
            # Forecast until December 2027
            last_date = df['time'].max()
            end_date = pd.to_datetime("2027-12-01")
            # Calculating number of months needed
            num_months = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)
            
            future_dates = pd.date_range(start=last_date, periods=num_months + 1, freq='MS')[1:]
            
            last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
            future_preds = []
            
            for _ in range(len(future_dates)):
                with torch.no_grad():
                    p = model(last_input)
                    val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                    future_preds.append(val)
                    # Recursive stability
                    last_input = last_input 

        # --- Professional Visualization ---
        fig = go.Figure()
        
        # Zones
        fig.add_hrect(y0=0.5, y1=2.5, fillcolor="red", opacity=0.08, line_width=0, annotation_text="El Niño Phase")
        fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor="blue", opacity=0.08, line_width=0, annotation_text="La Niña Phase")
        
        # Historical
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['nino34_anom'], 
            name="Historical Data", 
            line=dict(color='gray', width=1.5)
        ))
        
        # Future Forecast (2027 तक)
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_preds, 
            name="Future Forecast", 
            mode='lines',
            line=dict(color='orange', width=3)
        ))

        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            height=600,
            hovermode="x unified",
            xaxis=dict(title="Timeline", rangeslider=dict(visible=True)),
            yaxis=dict(title="Oceanic Anomaly Index", range=[-2.5, 2.5])
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Yearly Summary Table ---
        st.subheader("🗓️ Forecast Summary (Through 2027)")
        res_df = pd.DataFrame({
            "Month/Year": future_dates.strftime('%b %Y'), 
            "Index Value": [round(float(x), 2) for x in future_preds]
        })
        
        def check_status(v):
            if v > 0.5: return "🔴 Warning: El Niño Conditions"
            if v < -0.5: return "🔵 Favorable: La Niña Conditions"
            return "🟢 Stable: Neutral"
            
        res_df['Condition'] = res_df['Index Value'].apply(check_status)
        st.dataframe(res_df, use_container_width=True)

except Exception as e:
    st.error(f"System Error: {e}")
