import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Dummy data historis
np.random.seed(42)
date_range = pd.date_range(start="2024-01-01", end="2024-12-31", freq='D')
cpu_usage = np.random.normal(loc=50, scale=10, size=len(date_range))
cpu_usage = np.clip(cpu_usage, 10, 100)
memory_usage = np.random.uniform(0.5, 3.5, len(date_range))
request_count = np.random.randint(1000, 10000, len(date_range))

dummy_data = pd.DataFrame({
    'datetime': date_range,
    'cpu_usage': cpu_usage,
    'memory_usage': memory_usage,
    'request_count': request_count
})

# Data referensi Azure App Service Plans
azure_tiers = pd.DataFrame([
    {"Name": "Free F1", "CPU": "60", "Memory": "N/A", "Scale": 1},
    {"Name": "Basic B1", "CPU": "100", "Memory": "1.75", "Scale": 3},
    {"Name": "Basic B2", "CPU": "100", "Memory": "3.5", "Scale": 3},
    {"Name": "Basic B3", "CPU": "100", "Memory": "7", "Scale": 3},
    {"Name": "Premium v2 P1V2", "CPU": "210", "Memory": "3.5", "Scale": 30},
    {"Name": "Premium v2 P2V2", "CPU": "210", "Memory": "7", "Scale": 30},
    {"Name": "Premium v2 P3V2", "CPU": "210", "Memory": "14", "Scale": 30},
])

# Pastikan semua kolom `CPU` dan `Memory` adalah angka
def clean_column(column):
    return pd.to_numeric(column, errors='coerce').fillna(0)  # Ganti nilai yang tidak valid dengan 0

azure_tiers['CPU'] = clean_column(azure_tiers['CPU'])
azure_tiers['Memory'] = clean_column(azure_tiers['Memory'])

# Data untuk Prophet
prophet_data_cpu = dummy_data.rename(columns={'datetime': 'ds', 'cpu_usage': 'y'})

# Model Prophet untuk CPU usage
model_cpu = Prophet()
model_cpu.fit(prophet_data_cpu)

# Prediksi CPU usage untuk 30 hari ke depan
future_cpu = model_cpu.make_future_dataframe(periods=30)
forecast_cpu = model_cpu.predict(future_cpu)

# Tambahkan memory_usage dan request_count
forecast_cpu['memory_usage'] = dummy_data['memory_usage'].mean()
forecast_cpu['request_count'] = dummy_data['request_count'].mean()

# Fungsi untuk mapping ke tier Azure
def map_to_tier(row, tiers):
    for _, tier in tiers.iterrows():
        # Bandingkan hanya jika Memory adalah angka
        if row['yhat'] <= tier['CPU'] and row['memory_usage'] <= tier['Memory']:
            return tier['Name']
    return "Upgrade Needed"

# Terapkan mapping tier
forecast_cpu['tier'] = forecast_cpu.apply(lambda row: map_to_tier(row, azure_tiers), axis=1)

# Tampilkan hasil prediksi dan rekomendasi tier
forecast_result = forecast_cpu[['ds', 'yhat', 'memory_usage', 'request_count', 'tier']].tail(30)

# Visualisasi hasil prediksi CPU
fig = model_cpu.plot(forecast_cpu)
plt.title("CPU Usage Prediction")
plt.xlabel("Date")
plt.ylabel("CPU Usage (%)")
plt.show()

# Tampilkan hasil akhir
print(forecast_result)
