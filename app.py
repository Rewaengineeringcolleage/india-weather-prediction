import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# 1. Page Configuration (No AI mentions)
st.set_page_config(page_title="India Weather Forecast System", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("### 1970 to 2027 Climate Outlook (Historical & Future Forecast)")

# 2. Internal Data Loading
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
        
        # --- Processing Core ---
        features = ['uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        # Model Training Dataset
        dataset = {
            'train_input': torch.tensor(X_s[:-12], dtype=torch.float32),
            'train_label': torch.tensor(y_s[:-12], dtype=torch.float32),
            'test_input': torch.tensor(X_s[-12:], dtype=torch.float32),
            'test_label': torch.tensor(y_s[-12:], dtype=torch.float32)
        }
        
        with st.spinner("Processing long-term climate cycles..."):
            # Using KAN architecture for trend analysis
            model = KAN(width=[3, 3, 1], grid=3, k=3)
            model.fit(dataset, steps=5) 
            
            # Setting Timeline to December 2027
            last_date = df['time'].max()
            end_date = pd.to_datetime("2027-12-01")
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

        # --- Dashboard Visualization ---
        fig = go.Figure()
        
        # Adding Colored Status Zones
        fig.add_hrect(y0=0.5, y1=2.5, fillcolor="red", opacity=0.08, line_width=0, annotation_text="El Niño Region (Dry)")
        fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor="blue", opacity=0.08, line_width=0, annotation_text="La Niña Region (Wet)")
        fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="green", opacity=0.05, line_width=0, annotation_text="Neutral Region")
        
        # Plotting Historical Data
        fig.add_trace(go.Scatter(x=df['time'], y=df['nino34_anom'], name="Historical Record", line=dict(color='gray', width=1.5)))
        
        # Plotting Future Forecast (until 2027)
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Calculated Forecast", mode='lines', line=dict(color='orange', width=3)))

        fig.add_hline(y=0, line_color="black", line_width=1)

        fig.update_layout(
            height=600,
            hovermode="x unified",
            xaxis=dict(title="Timeline", rangeslider=dict(visible=True)),
            yaxis=dict(title="Oceanic Anomaly Index", range=[-2.5, 2.5])
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Status Classification ---
        st.subheader("🗓️ Forecast Summary & Indian Monsoon Impact")
        
        # Displaying the very next month's status prominently
        latest_val = future_preds[0]
        if latest_val > 0.5:
            st.error(f"Current Forecast: El Niño detected ({latest_val:.2f}). Potential for below-normal rainfall.")
        elif latest_val < -0.5:
            st.success(f"Current Forecast: La Niña detected ({latest_val:.2f}). Favorable for strong monsoon rainfall.")
        else:
            st.info(f"Current Forecast: Neutral conditions ({latest_val:.2f}). Expect normal seasonal patterns.")

        # Full Table Breakdown
        res_df = pd.DataFrame({
            "Month/Year": future_dates.strftime('%b %Y'), 
            "Index Value": [round(float(x), 2) for x in future_preds]
        })
        
        def classify_climate(v):
            if v > 0.5: return "🔴 El Niño (Risk of Drought)"
            if v < -0.5: return "🔵 La Niña (Good Monsoon)"
            return "🟢 Neutral (Normal Rain)"
            
        res_df['Climate Condition'] = res_df['Index Value'].apply(classify_climate)
        st.dataframe(res_df, use_container_width=True)

except Exception as e:
    st.error(f"System Load Error: {e}")
