import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- CHART 1: 2026 Monthly Predictions ---
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# ONI Index Prediction (Approximate values based on current 2026 outlook)
oni_2026 = [-0.6, -0.5, -0.2, 0.1, 0.3, 0.6, 0.8, 1.1, 1.3, 1.5, 1.6, 1.5]

plt.figure(figsize=(12, 5))
colors = ['blue' if x < -0.5 else 'red' if x > 0.5 else 'gray' for x in oni_2026]
plt.bar(months, oni_2026, color=colors, alpha=0.7)
plt.axhline(0.5, color='red', linestyle='--', linewidth=0.8, label='El Niño Threshold')
plt.axhline(-0.5, color='blue', linestyle='--', linewidth=0.8, label='La Niña Threshold')
plt.title('Monthly ENSO Prediction for 2026')
plt.ylabel('ONI Index (°C)')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()

# --- CHART 2: Flag Chart (1960 - 2030) ---
years = np.arange(1960, 2031)
# 0 = Neutral, 1 = El Niño, -1 = La Niña (Simplified for visualization)
events = np.zeros(len(years))

# Adding major historical & predicted markers
la_nina_years = [1964, 1970, 1973, 1975, 1988, 1998, 1999, 2007, 2010, 2020, 2021, 2022, 2025]
el_nino_years = [1965, 1972, 1982, 1987, 1991, 1997, 2002, 2009, 2015, 2023, 2026, 2027, 2029]

for i, y in enumerate(years):
    if y in la_nina_years: events[i] = -1
    elif y in el_nino_years: events[i] = 1

plt.figure(figsize=(15, 3))
plt.fill_between(years, events, color='gray', alpha=0.2) # Baseline
plt.bar(years[events==1], 1, color='red', label='El Niño')
plt.bar(years[events==-1], -1, color='blue', label='La Niña')

plt.title('ENSO Timeline Flag Chart (1960 - 2030)')
plt.yticks([-1, 0, 1], ['La Niña', 'Neutral', 'El Niño'])
plt.xticks(np.arange(1960, 2031, 5))
plt.grid(axis='x', alpha=0.2)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
