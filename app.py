import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ===== 1. Загружаем данные =====
df = pd.read_csv("data.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

# ===== 2. Обучение ARIMA =====
model = ARIMA(df["patients"], order=(2,1,2))
model_fit = model.fit()

# ===== 3. Прогноз на 7 дней вперёд =====
forecast = model_fit.forecast(steps=7)

# ===== 4. Сохраняем график =====
plt.figure(figsize=(10,6))
plt.plot(df.index, df["patients"], label="Исторические данные", color="blue")
plt.plot(forecast.index, forecast, label="Прогноз", color="red", linestyle="dashed")
plt.xlabel("Дата")
plt.ylabel("Пациенты")
plt.title("Прогноз нагрузки на больницу (ARIMA)")
plt.legend()
plt.grid(True)
plt.savefig("forecast.png")

print("Прогноз успешно построен! Смотри файл forecast.png")
