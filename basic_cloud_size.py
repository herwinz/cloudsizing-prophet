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
    {"Type": "Dev/test", "Name": "Free F1", "CPU": "60 minutes/day", "Memory": "N/A", "Remote Storage (GB)": 1, "Scale": "N/A", "SLA": "N/A", "Cost per hour (USD)": "Free"},
    {"Type": "Dev/test", "Name": "Basic B1", "CPU": 100, "Memory": 1.75, "Remote Storage (GB)": 10, "Scale": 3, "SLA": 99.95, "Cost per hour (USD)": 0.018},
    {"Type": "Dev/test", "Name": "Basic B2", "CPU": 100, "Memory": 3.5, "Remote Storage (GB)": 10, "Scale": 3, "SLA": 99.95, "Cost per hour (USD)": 0.036},
    {"Type": "Dev/test", "Name": "Basic B3", "CPU": 100, "Memory": 7, "Remote Storage (GB)": 10, "Scale": 3, "SLA": 99.95, "Cost per hour (USD)": 0.071},
    {"Type": "Production", "Name": "Premium v3 P5mv3", "CPU": 195, "Memory": 32, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 3.418},
    {"Type": "Production", "Name": "Premium v3 P4mv3", "CPU": 195, "Memory": 16, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 1.709},
    {"Type": "Production", "Name": "Premium v3 P3mv3", "CPU": 195, "Memory": 8, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 0.854},
    {"Type": "Production", "Name": "Premium v3 P2mv3", "CPU": 195, "Memory": 4, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 0.427},
    {"Type": "Production", "Name": "Premium v3 P1mv3", "CPU": 195, "Memory": 2, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 0.214},
    {"Type": "Production", "Name": "Premium v3 P0mv3", "CPU": 195, "Memory": 1, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 0.089},
    {"Type": "Legacy", "Name": "Standard S3", "CPU": 100, "Memory": 4, "Remote Storage (GB)": 50, "Scale": 10, "SLA": 99.95, "Cost per hour (USD)": 0.38},
    {"Type": "Legacy", "Name": "Standard S2", "CPU": 100, "Memory": 3.5, "Remote Storage (GB)": 50, "Scale": 10, "SLA": 99.95, "Cost per hour (USD)": 0.19},
    {"Type": "Legacy", "Name": "Standard S1", "CPU": 100, "Memory": 1.75, "Remote Storage (GB)": 50, "Scale": 10, "SLA": 99.95, "Cost per hour (USD)": 0.095},
    {"Type": "Legacy", "Name": "Premium v2 P3V2", "CPU": 210, "Memory": 14, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 0.462},
    {"Type": "Legacy", "Name": "Premium v2 P2V2", "CPU": 210, "Memory": 7, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 0.231},
    {"Type": "Legacy", "Name": "Premium v2 P1V2", "CPU": 210, "Memory": 3.5, "Remote Storage (GB)": 250, "Scale": 30, "SLA": 99.95, "Cost per hour (USD)": 0.115}
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
