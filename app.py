import streamlit as st
import pandas as pd
import numpy as np
import torch
from kan import KAN
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# 1. Page Setup (No mentions of AI)
st.set_page_config(page_title="Climate Prediction System", layout="wide")
st.title("🇮🇳 India ENSO & Monsoon Predictor")
st.markdown("### 1970 to 2030 Climate Outlook (Long-Term Cycle Forecast)")

@st.cache_data
def load_data():
    file_path = "enso_all_merged_data (1) FINALE.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower()
    df['time'] = pd.to_datetime(df['time'])
    # Adding Month as a feature to understand seasonal cycles
    df['month_val'] = df['time'].dt.month
    return df[df['time'].dt.year >= 1970]

try:
    df = load_data()
    
    if st.button("📊 Generate 2030 Forecast Report"):
        
        # --- Data Prep with Seasonality ---
        # Features: Month, Wind(U), Wind(V), Pressure
        features = ['month_val', 'uwnd', 'vwnd', 'slp']
        X = df[features].values
        y = df['nino34_anom'].values.reshape(-1, 1)
        
        scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
        X_s = scaler_x.fit_transform(X)
        y_s = scaler_y.fit_transform(y)
        
        # Dataset setup
        dataset = {
            'train_input': torch.tensor(X_s, dtype=torch.float32),
            'train_label': torch.tensor(y_s, dtype=torch.float32),
            'test_input': torch.tensor(X_s[-5:], dtype=torch.float32),
            'test_label': torch.tensor(y_s[-5:], dtype=torch.float32)
        }
        
        with st.spinner("Analyzing climate cycles and historical patterns..."):
            # Forecasting Model
            model = KAN(width=[4, 3, 1], grid=3, k=3)
            model.fit(dataset, steps=8) 
            
            # Prediction loop until Dec 2030
            last_date = df['time'].max()
            end_date = pd.to_datetime("2030-12-01")
            num_months = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)
            future_dates = pd.date_range(start=last_date, periods=num_months + 1, freq='MS')[1:]
            
            future_preds = []
            # Start with the last known physical state
            current_input = X_s[-1:].copy()
            
            for d in future_dates:
                with torch.no_grad():
                    # Update the 'Month' feature for each step so it's not a flat line
                    m_scaled = (d.month - 1) / 11.0 
                    current_input[0][0] = m_scaled 
                    
                    p = model(torch.tensor(current_input, dtype=torch.float32))
                    val = scaler_y.inverse_transform(p.detach().numpy())[0][0]
                    future_preds.append(val)

        # --- Dashboard Visualization ---
        fig = go.Figure()
        
        # Condition Zones
        fig.add_hrect(y0=0.5, y1=3.0, fillcolor="red", opacity=0.1, line_width=0, annotation_text="El Niño (Hot/Dry)")
        fig.add_hrect(y0=-3.0, y1=-0.5, fillcolor="blue", opacity=0.1, line_width=0, annotation_text="La Niña (Strong Monsoon)")
        fig.add_hrect(y0=-0.5, y1=0.5, fillcolor="green", opacity=0.05, line_width=0, annotation_text="Neutral (Normal)")
        
        # Historical Record
        fig.add_trace(go.Scatter(x=df['time'], y=df['nino34_anom'], name="Past Observations", line=dict(color='gray', width=1)))
        
        # Future Trend (Orange Line)
        fig.add_trace(go.Scatter(x=future_dates, y=future_preds, name="Predicted Trend to 2030", line=dict(color='orange', width=3)))

        fig.add_hline(y=0, line_color="black", line_width=1)

        # Clean X-axis with 2-year interval
        fig.update_layout(
            height=600,
            hovermode="x unified",
            xaxis=dict(title="Year", dtick="M24", tickformat="%Y", rangeslider=dict(visible=True)),
            yaxis=dict(title="Ocean Anomaly Index (Nino 3.4)", range=[-2.5, 2.5]),
            legend=dict(orientation="h", y=1.1, x=1, xanchor="right")
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Analysis Summary ---
        st.subheader("📋 Predicted Climate Conditions (2025-2030)")
        
        # Yearly averages for quick view
        res_temp = pd.DataFrame({"Date": future_dates, "Val": future_preds})
        summary_cols = st.columns(6)
        years = [2025, 2026, 2027, 2028, 2029, 2030]
        
        for i, year in enumerate(years):
            yearly_val = res_temp[res_temp['Date'].dt.year == year]['Val'].mean()
            if not np.isnan(yearly_val):
                status = "Neutral"
                if yearly_val > 0.5: status = "El Niño"
                elif yearly_val < -0.5: status = "La Niña"
                summary_cols[i].metric(f"Year {year}", f"{yearly_val:.2f}", status)

        # Detailed Month-by-Month Table
        st.subheader("🗓️ Monthly Forecast Breakdown")
        res_df = pd.DataFrame({
            "Month/Year": future_dates.strftime('%B %Y'), 
            "Index Value": [round(float(x), 2) for x in future_preds]
        })
        
        def get_impact(v):
            if v > 0.5: return "🔴 El Niño (Risk of Drought)"
            if v < -0.5: return "🔵 La Niña (Heavy Monsoon)"
            return "🟢 Neutral (Normal Rain)"
            
        res_df['Climate Status'] = res_df['Index Value'].apply(get_impact)
        st.dataframe(res_df, use_container_width=True)

except Exception as e:
    st.error(f"Error loading prediction: {e}")

import seaborn as sns
import matplotlib.pyplot as plt

# Sidebar ya Main page par Heatmap ka option
if st.checkbox("Show Correlation Heatmap"):
    st.subheader("Feature Correlation Analysis")
    fig_corr, ax_corr = plt.subplots()
    # Sirf numerical columns ka correlation
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax_corr)
    st.pyplot(fig_corr)

# Sunspot Accuracy Graph
if st.checkbox("Show Sunspot Analysis"):
    st.subheader("Sunspot Activity vs Nino 3.4")
    # Plotting sunspot data from your CSV
    st.line_chart(df.set_index('time')['sunspot'])
