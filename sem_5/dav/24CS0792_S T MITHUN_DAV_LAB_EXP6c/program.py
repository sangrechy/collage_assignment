"""
AIM: Time series analysis of glucose levels - trend/seasonality decomposition,
moving average smoothing, and ARIMA forecasting.
"""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA

diabetes_data = pd.read_csv("../data/diabetes9.csv")
print(diabetes_data.head())

plt.figure(figsize=(12, 5))
plt.plot(diabetes_data["Glucose"], label="Glucose Level", color="blue")
plt.xlabel("Index"); plt.ylabel("Glucose Level")
plt.title("Time Series of Glucose Levels")
plt.legend()
plt.savefig("timeseries_raw.png", dpi=120, bbox_inches="tight")
plt.close()

decomposition = seasonal_decompose(diabetes_data["Glucose"], model="additive", period=30)
fig, axes = plt.subplots(3, 1, figsize=(12, 8))
decomposition.trend.plot(ax=axes[0], title="Trend Component")
decomposition.seasonal.plot(ax=axes[1], title="Seasonal Component")
decomposition.resid.plot(ax=axes[2], title="Residual Component")
plt.tight_layout()
plt.savefig("decomposition.png", dpi=120, bbox_inches="tight")
plt.close()

diabetes_data["Glucose_MA"] = diabetes_data["Glucose"].rolling(window=7).mean()
plt.figure(figsize=(12, 5))
plt.plot(diabetes_data["Glucose"], label="Original", alpha=0.5)
plt.plot(diabetes_data["Glucose_MA"], label="7-day Moving Average", color="red")
plt.legend(); plt.title("Moving Average Smoothing")
plt.savefig("moving_average.png", dpi=120, bbox_inches="tight")
plt.close()

train_size = int(len(diabetes_data) * 0.8)
train = diabetes_data["Glucose"][:train_size]
test = diabetes_data["Glucose"][train_size:]

model = ARIMA(train, order=(5, 1, 0))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=len(test))

plt.figure(figsize=(12, 5))
plt.plot(range(len(test)), test, label="Actual", color="blue")
plt.plot(range(len(test)), forecast, label="Forecast", color="red")
plt.xlabel("Index"); plt.ylabel("Glucose Level")
plt.title("ARIMA Model Forecasting")
plt.legend()
plt.savefig("arima_forecast.png", dpi=120, bbox_inches="tight")
plt.close()

print("Saved timeseries_raw.png, decomposition.png, moving_average.png, arima_forecast.png")
