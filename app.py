import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Page Configuration (Professional Look)
st.set_page_config(page_title="Climate Prediction 2030", layout="wide")
st.title("🇮🇳 India Monsoon & ENSO Predictor")
st.markdown("### 1970 - 2030: Historical Trends & Future Climate Outlook")

@st.cache_data
def load_data():
    file_path = "enso_all_merged_data (1) FINALE.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    # Adding Time-Cycle features to force Neutral/El Nino changes
    df['month'] = df['time'].dt.month
    return df[df['time'].dt.year >= 1970]

try:
    df = load_data()
    
    tab1, tab2, tab3 = st.tabs(["📊 2030 Forecast", "🔥 Correlation Heatmap", "☀️ Sunspot Analysis"])

    with tab1:
        if st.button("📈 Generate Full Report (1970 - 2030)"):
            # Features: Month, Wind, Pressure, Sunspots
            features = ['month', 'uwnd', 'vwnd', 'slp', 'sunspot']
            X = df[features].values
            y = df['nino34_anom'].values.reshape(-1, 1)
            
            scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_s = scaler_x.fit_transform(X)
            y_s = scaler_y.fit_transform(y)
            
            # Model Setup
            model = KAN(width=[5, 3, 1], grid=3, k=3)
            dataset = {'train_input': torch.tensor(X_s, dtype=torch.float32), 
                       'train_label': torch.tensor(y_s, dtype=torch.float32),
                       'test_input': torch.tensor(X_s[-5:], dtype=torch.float32),
                       'test_label': torch.tensor(y_s[-5:], dtype=torch.float32)}
            
            with st.spinner("Calculating 60-year climate cycles..."):
                model.fit(dataset, steps=10) # Higher steps for better accuracy
                
                # Predicting until Dec 2030
                future_dates = pd.date_range(start=df['time'].max(), periods=61, freq='MS')[1:]
                future_preds = []
                
                # Dynamic forecasting to avoid flat 'La Nina' line
                last_input = X_s[-1:].copy()
                for d in future_dates:
                    last_input[0][0] = (d.month - 1) / 11.0 # Update month feature
                    p = model(torch.tensor(last_input, dtype=torch.float32))
                    val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                    future_preds.append(val)

            # --- CLEAN GRAPH DESIGN ---
            fig = go.Figure()
            
            # 1. Defined Background Zones
            fig.add_hrect(y0=0.5, y1=3.0, fillcolor="red", opacity=0.1, line_width=0, annotation_text="El Niño (Dry)")
            fig.add_hrect(y0=-3.0, y1=-0.5, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="La Niña (Wet)")
            fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="green", opacity=0.05, line_width=0, annotation_text="Neutral")

            # 2. Historical & Future Lines
            fig.add_trace(go.Scatter(x=df['time'], y=df['nino34_anom'], name="Historical", line=dict(color='gray', width=1)))
            fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="2030 Forecast", line=dict(color='orange', width=3)))

            # 3. CLEAN X-AXIS (Months and Years)
            fig.update_layout(
                height=600,
                hovermode="x unified",
                xaxis=dict(
                    title="Timeline",
                    tickformat="%b %Y", # Shows 'Jan 2025'
                    dtick="M24",        # Label every 2 years
                    rangeslider=dict(visible=True), # Zoom to see months
                    gridcolor='lightgray'
                ),
                yaxis=dict(title="Index Value", range=[-2.5, 2.5], gridcolor='lightgray'),
                plot_bgcolor='white'
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # 4. Detailed Results Table
            st.subheader("🗓️ Monthly Breakdown (Forecast)")
            res_df = pd.DataFrame({"Month/Year": future_dates.strftime('%B %Y'), "Index": [round(float(x), 2) for x in future_preds]})
            
            def get_status(v):
                if v > 0.5: return "🔴 El Niño"
                if v < -0.5: return "🔵 La Niña"
                return "🟢 Neutral"
            
            res_df['Condition'] = res_df['Index'].apply(get_status)
            st.dataframe(res_df, use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation Heatmap")
        fig_h, ax_h = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', ax=ax_h)
        st.pyplot(fig_h)

    with tab3:
        st.subheader("Sunspot Activity vs Nino 3.4")
        st.line_chart(df.set_index('time')['sunspot'])

except Exception as e:
    st.error(f"Error: {e}")
