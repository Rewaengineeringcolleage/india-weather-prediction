import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Streamlit Page Configuration
st.set_page_config(page_title="Climate Prediction Dashboard", layout="wide")

st.title("🌍 Climate Event Prediction: El Niño & La Niña")
st.markdown("""
Aapka ye dashboard 1960 se 2030 tak ke climate trends aur 2026 ka monthly forecast dikhata hai.
""")

# --- DATA PREPARATION (Static for Visualization) ---
years = np.arange(1960, 2031)
la_nina_years = [1964, 1970, 1973, 1975, 1988, 1998, 1999, 2007, 2010, 2020, 2021, 2022, 2025]
el_nino_years = [1965, 1972, 1982, 1987, 1991, 1997, 2002, 2009, 2015, 2023, 2026, 2027, 2029]

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# Simulated ONI Index for 2026 (Transition from La Niña to El Niño)
oni_2026 = [-0.6, -0.5, -0.2, 0.1, 0.3, 0.6, 0.8, 1.1, 1.3, 1.5, 1.6, 1.5]

# --- LAYOUT: TWO COLUMNS ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📅 2026 Monthly Prediction")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Color logic: Red for El Nino (>0.5), Blue for La Nina (<-0.5)
    colors = ['#3498db' if x < -0.5 else '#e74c3c' if x > 0.5 else '#bdc3c7' for x in oni_2026]
    
    ax1.bar(months, oni_2026, color=colors, edgecolor='black', alpha=0.8)
    ax1.axhline(0.5, color='red', linestyle='--', linewidth=1, label='El Niño Threshold')
    ax1.axhline(-0.5, color='blue', linestyle='--', linewidth=1, label='La Niña Threshold')
    ax1.set_ylabel("ONI Index (°C)")
    ax1.set_title("Forecast for the Year 2026")
    ax1.legend()
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    st.pyplot(fig1)

with col2:
    st.subheader("🚩 Historical & Future Timeline (1960-2030)")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    events = np.zeros(len(years))
    for i, y in enumerate(years):
        if y in la_nina_years: events[i] = -1
        elif y in el_nino_years: events[i] = 1
    
    # Drawing the Flag Chart
    ax2.fill_between(years, events, color='gray', alpha=0.1)
    ax2.bar(years[events==1], 1, color='#e74c3c', label='El Niño', width=0.8)
    ax2.bar(years[events==-1], -1, color='#3498db', label='La Niña', width=0.8)
    
    ax2.set_yticks([-1, 0, 1])
    ax2.set_yticklabels(['La Niña', 'Neutral', 'El Niño'])
    ax2.set_xticks(np.arange(1960, 2031, 10))
    ax2.set_title("Event Classification (1960 - 2030)")
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    st.pyplot(fig2)

# --- ADDITIONAL INFO ---
st.divider()
st.info("**Model Note:** Yeh predictions ENSO (El Niño-Southern Oscillation) patterns par based hain. Blue bars thandak (La Niña) ko darshati hain aur Red bars garmi (El Niño) ko.")
