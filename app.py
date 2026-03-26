import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Climate Dashboard 2030", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")

@st.cache_data
def load_data():
    file_path = "enso_all_merged_data (1) FINALE.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    return df[df['time'].dt.year >= 1970]

try:
    df = load_data()
    tab1, tab2, tab3 = st.tabs(["📊 2030 Forecast", "🔥 Correlation Heatmap", "🔆 Sunspot Trends"])

    with tab1:
        if st.button("🚀 Run 2030 Forecast"):
            # Model Logic
            features = ['uwnd', 'vwnd', 'slp', 'sunspot']
            X = df[features].values
            y = df['nino34_anom'].values.reshape(-1, 1)
            
            scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
            X_s = scaler_x.fit_transform(X)
            y_s = scaler_y.fit_transform(y)
            
            model = KAN(width=[4, 3, 1], grid=3, k=3)
            dataset = {'train_input': torch.tensor(X_s[:-12], dtype=torch.float32), 
                       'train_label': torch.tensor(y_s[:-12], dtype=torch.float32),
                       'test_input': torch.tensor(X_s[-12:], dtype=torch.float32),
                       'test_label': torch.tensor(y_s[-12:], dtype=torch.float32)}
            
            with st.spinner("Processing..."):
                model.fit(dataset, steps=5)
                
                # Forecast 2024 to 2030
                future_dates = pd.date_range(start=df['time'].max(), periods=73, freq='MS')[1:]
                future_preds = [scaler_y.inverse_transform(model(torch.tensor(X_s[-1:], dtype=torch.float32)).detach().numpy())[0][0] for _ in range(72)]
                
                fig = go.Figure()
                
                # Background Zones
                fig.add_hrect(y0=0.5, y1=2.5, fillcolor="red", opacity=0.1, annotation_text="El Niño")
                fig.add_hrect(y0=-2.5, y1=-0.5, fillcolor="blue", opacity=0.1, annotation_text="La Niña")
                
                # Data Lines
                fig.add_trace(go.Scatter(x=df['time'], y=df['nino34_anom'], name="Past Record", line=dict(color='gray')))
                fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Future Forecast", line=dict(color='orange', width=3)))

                # --- X-AXIS FIX FOR MONTHS ---
                fig.update_layout(
                    height=600,
                    hovermode="x unified",
                    xaxis=dict(
                        title="Timeline (Month & Year)",
                        type="date",
                        tickformat="%b %Y", # Isse 'Jan 2025' jaisa format dikhega
                        dtick="M24",        # Har 24 mahine (2 saal) par label dikhega
                        tickangle=-45,      # Labels thode tede honge taaki jagah bane
                        rangeslider=dict(visible=True) # Isse aap zoom karke mahine dekh sakte hain
                    ),
                    yaxis=dict(title="Nino 3.4 Index", range=[-2.5, 2.5])
                )
                
                st.plotly_chart(fig, use_container_width=True)

                # Monthly Table (Yahan har mahina saaf dikhega)
                st.subheader("🗓️ Detailed Monthly Data (2025-2030)")
                res_df = pd.DataFrame({
                    "Date": future_dates.strftime('%B %Y'),
                    "Index": [round(float(x), 2) for x in future_preds]
                })
                st.dataframe(res_df, use_container_width=True)

    with tab2:
        # Heatmap Code
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', ax=ax_heat)
        st.pyplot(fig_heat)

    with tab3:
        # Sunspot Code
        st.line_chart(df.set_index('time')['sunspot'])

except Exception as e:
    st.error(f"Error: {e}")
