# ============================================================
# Time Series Analysis - Air Passengers Forecasting
# File: Time_Series_Analysis.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error


def main():

    # -------------------------
    # 1) LOAD DATA
    # -------------------------
    df = pd.read_csv("AirPassengers.csv")

    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)

    print(df.head())

    # -------------------------
    # 2) PLOT TIME SERIES
    # -------------------------
    plt.figure()
    plt.plot(df["Passengers"])
    plt.title("Air Passenger Trend")
    plt.xlabel("Year")
    plt.ylabel("Passengers")
    plt.show()

    # -------------------------
    # 3) DECOMPOSITION
    # -------------------------
    decomposition = seasonal_decompose(df["Passengers"], model="additive", period=12)
    decomposition.plot()
    plt.show()

    # -------------------------
    # 4) TRAIN TEST SPLIT
    # -------------------------
    train = df[:'1950-06']
    test = df['1950-07':]

    # -------------------------
    # 5) ARIMA MODEL
    # -------------------------
    model = ARIMA(train["Passengers"], order=(1, 1, 1))
    model_fit = model.fit()

    print(model_fit.summary())

    # -------------------------
    # 6) FORECAST
    # -------------------------
    forecast = model_fit.forecast(steps=len(test))

    # -------------------------
    # 7) EVALUATION
    # -------------------------
    rmse = np.sqrt(mean_squared_error(test["Passengers"], forecast))
    print("RMSE:", rmse)

    # -------------------------
    # 8) PLOT FORECAST
    # -------------------------
    plt.figure()
    plt.plot(train.index, train["Passengers"], label="Train")
    plt.plot(test.index, test["Passengers"], label="Actual")
    plt.plot(test.index, forecast, label="Forecast")
    plt.legend()
    plt.title("ARIMA Forecast")
    plt.show()

    print("\n✅ Time Series Analysis Completed!")


if __name__ == "__main__":
    main()