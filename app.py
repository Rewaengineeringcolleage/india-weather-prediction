import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Climate Prediction System", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("### 1970 to 2030 Climate Outlook (Historical & Future Forecast)")

# 2. Loading Data
@st.cache_data
def load_data():
    file_path = "enso_all_merged_data (1) FINALE.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    # Filtering from 1970 as requested
    return df[df['time'].dt.year >= 1970]

try:
    df = load_data()
    
    if st.button("📊 Generate Long-Term Forecast (1970-2030)"):
        
        # --- Processing Core ---
        features = ['uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        dataset = {
            'train_input': torch.tensor(X_s[:-12], dtype=torch.float32),
            'train_label': torch.tensor(y_s[:-12], dtype=torch.float32),
            'test_input': torch.tensor(X_s[-12:], dtype=torch.float32),
            'test_label': torch.tensor(y_s[-12:], dtype=torch.float32)
        }
        
        with st.spinner("Analyzing multi-decadal climate cycles until 2030..."):
            model = KAN(width=[3, 3, 1], grid=3, k=3)
            model.fit(dataset, steps=5) 
            
            # Forecast Range: Current to December 2030
            last_date = df['time'].max()
            end_date = pd.to_datetime("2030-12-01")
            num_months = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)
            
            future_dates = pd.date_range(start=last_date, periods=num_months + 1, freq='MS')[1:]
            last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
            future_preds = []
            
            for _ in range(len(future_dates)):
                with torch.no_grad():
                    p = model(last_input)
                    val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                    future_preds.append(val)
                    last_input = last_input 

        # --- Clean & Professional Visualization ---
        fig = go.Figure()
        
        # Clear Background Color Zones
        fig.add_hrect(y0=0.5, y1=2.5, fillcolor="red", opacity=0.07, line_width=0, annotation_text="El Niño (Dry/Warm)")
        fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor="blue", opacity=0.07, line_width=0, annotation_text="La Niña (Wet/Cool)")
        fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="green", opacity=0.04, line_width=0, annotation_text="Normal/Neutral")
        
        # Historical Record
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['nino34_anom'], 
            name="Historical Data", 
            line=dict(color='rgba(100,100,100,0.6)', width=1.5),
            hoverinfo="x+y"
        ))
        
        # Future Forecast (2030 तक)
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_preds, 
            name="Future Forecast", 
            mode='lines',
            line=dict(color='orange', width=3.5),
            hoverinfo="x+y"
        ))

        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            height=650,
            hovermode="x unified",
            xaxis=dict(
                title="Yearly Timeline",
                rangeslider=dict(visible=True), # Zoom into specific years
                type="date",
                tickformat="%Y"
            ),
            yaxis=dict(title="Oceanic Anomaly Index", range=[-2.5, 2.5]),
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Summary Section ---
        st.subheader("🗓️ Long-Term Outlook Summary")
        
        # Monthly Classification Logic
        res_df = pd.DataFrame({
            "Month/Year": future_dates.strftime('%B %Y'), 
            "Index": [round(float(x), 2) for x in future_preds]
        })
        
        def classify(v):
            if v > 0.5: return "🔴 El Niño Conditions"
            if v < -0.5: return "🔵 La Niña Conditions"
            return "🟢 Normal/Neutral"
            
        res_df['Climate Status'] = res_df['Index'].apply(classify)
        
        # Searchable Table
        st.dataframe(res_df, use_container_width=True)

except Exception as e:
    st.error(f"Error loading system: {e}")
