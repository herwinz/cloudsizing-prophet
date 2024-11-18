# install2 pake python3 -m pip install prophet 
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt

# Membuat dummy data (historical data usage)
np.random.seed(42)
date_range = pd.date_range(start="2024-01-01", end="2024-12-31", freq='D')
cpu_usage = np.random.normal(loc=50, scale=10, size=len(date_range))
cpu_usage = np.clip(cpu_usage, 10, 100)

dummy_data = pd.DataFrame({
    'datetime': date_range,
    'cpu_usage': cpu_usage
})

# Menyiapkan data untuk Prophet
prophet_data = dummy_data.rename(columns={'datetime': 'ds', 'cpu_usage': 'y'})

# Inisialisasi dan training model Prophet
model = Prophet()
model.fit(prophet_data)

# Membuat prediksi untuk 30 hari ke depan
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Menambahkan kolom prediksi ke dalam DataFrame asli
forecast_result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)

# Visualisasi hasil
fig = model.plot(forecast)
plt.title("CPU Usage Prediction")
plt.xlabel("Date")
plt.ylabel("CPU Usage (%)")
plt.show()

# Simulasi pemetaan prediksi ke tier Azure App Service Plan
def map_to_tier(cpu_usage):
    if cpu_usage < 40:
        return 'B1'
    elif 40 <= cpu_usage < 70:
        return 'B2'
    else:
        return 'B3'

forecast_result['tier'] = forecast_result['yhat'].apply(map_to_tier)

# Menampilkan hasil prediksi dan tier yang disarankan
print(forecast_result)

