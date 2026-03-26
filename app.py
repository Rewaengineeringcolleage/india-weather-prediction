import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="Climate Forecast 2030", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("### 1970 to 2030 Climate Outlook (Historical & Future Forecast)")

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
    
    if st.button("📊 Generate 2030 Long-Term Forecast"):
        
        # --- Data Processing ---
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
        
        with st.spinner("Processing climate cycles until 2030..."):
            model = KAN(width=[3, 3, 1], grid=3, k=3)
            model.fit(dataset, steps=5) 
            
            # Setting Timeline to December 2030
            last_date = df['time'].max()
            end_date = pd.to_datetime("2030-12-01")
            num_months = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)
            
            future_dates = pd.date_range(start=last_date, periods=num_months + 1, freq='MS')[1:]
            last_input = torch.tensor(X_s[-1:], dtype=torch.float32)
            future_preds = []
            
            # Generating forecast
            for _ in range(len(future_dates)):
                with torch.no_grad():
                    p = model(last_input)
                    val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                    future_preds.append(val)
                    last_input = last_input 

        # --- Clean Visual Dashboard ---
        fig = go.Figure()
        
        # Color Zones for Conditions
        fig.add_hrect(y0=0.5, y1=2.5, fillcolor="red", opacity=0.1, line_width=0, annotation_text="El Niño (Drought Risk)")
        fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="La Niña (High Rainfall)")
        fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="green", opacity=0.05, line_width=0, annotation_text="Neutral (Normal)")
        
        # Historical Trace
        fig.add_trace(go.Scatter(x=df['time'], y=df['nino34_anom'], name="Past Record", line=dict(color='gray', width=1.5)))
        
        # 2030 Forecast Trace
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Future Forecast (to 2030)", line=dict(color='orange', width=3)))

        fig.add_hline(y=0, line_color="black", line_width=1)

        # X-Axis setup for 2-year intervals
        fig.update_layout(
            height=600,
            hovermode="x unified",
            xaxis=dict(
                title="Year",
                tickformat="%Y",
                dtick="M24", # 24 months = 2 years interval
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(title="Oceanic Anomaly Index", range=[-2.5, 2.5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Status Cards ---
        st.subheader("📋 2030 Forecast Insights")
        
        # Show specific predictions for next 5 years (Summary)
        years_to_show = [2025, 2026, 2027, 2028, 2029, 2030]
        cols = st.columns(len(years_to_show))
        
        temp_df = pd.DataFrame({"Date": future_dates, "Val": future_preds})
        for i, year in enumerate(years_to_show):
            yearly_avg = temp_df[temp_df['Date'].dt.year == year]['Val'].mean()
            if pd.isna(yearly_avg): continue
            
            with cols[i]:
                if yearly_avg > 0.5:
                    st.metric(f"Year {year}", f"{yearly_avg:.2f}", "El Niño")
                elif yearly_avg < -0.5:
                    st.metric(f"Year {year}", f"{yearly_avg:.2f}", "-La Niña", delta_color="normal")
                else:
                    st.metric(f"Year {year}", f"{yearly_avg:.2f}", "Neutral", delta_color="off")

        # Detailed Table
        st.subheader("🗓️ Monthly Breakdown (2025 - 2030)")
        res_df = pd.DataFrame({
            "Month/Year": future_dates.strftime('%b %Y'), 
            "Index Value": [round(float(x), 2) for x in future_preds]
        })
        
        def classify(v):
            if v > 0.5: return "🔴 El Niño (Dry)"
            if v < -0.5: return "🔵 La Niña (Wet)"
            return "🟢 Neutral (Normal)"
            
        res_df['Condition'] = res_df['Index Value'].apply(classify)
        st.dataframe(res_df, use_container_width=True)

except Exception as e:
    st.error(f"System Load Error: {e}")
