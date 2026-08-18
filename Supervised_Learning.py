# ============================================================
# Supervised Learning Assignment - Diamond Price Prediction
# File: Supervised_Learning.py
# Dataset: diamonds.csv
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import joblib


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    # -------------------------
    # 1) LOAD DATA
    # -------------------------
    df = pd.read_csv("diamonds.csv")  # change to full path if needed

    # Drop index-like column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    print("Original shape:", df.shape)

    # -------------------------
    # 2) BASIC CLEANING
    # -------------------------
    df = df.drop_duplicates()

    # Handle missing values safely
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "string"]).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])

    # Remove invalid dimensions if present (x/y/z should be > 0)
    if set(["x", "y", "z"]).issubset(df.columns):
        df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

    print("After cleaning shape:", df.shape)

    # -------------------------
    # 3) ENCODE CATEGORICAL FEATURES
    # -------------------------
    df = pd.get_dummies(df, columns=["cut", "color", "clarity"], drop_first=True)

    # -------------------------
    # 4) SPLIT FEATURES / TARGET
    # -------------------------
    target = "price"
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("X_train:", X_train.shape, "| X_test:", X_test.shape)

    # -------------------------
    # 5) FEATURE SCALING
    # -------------------------
    # Scale only numeric columns that exist
    scaler = StandardScaler()

    numeric_features = [c for c in ["carat", "depth", "table", "x", "y", "z"] if c in X_train.columns]

    # Fit on train, transform both train and test
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numeric_features] = scaler.fit_transform(X_train_scaled[numeric_features])
    X_test_scaled[numeric_features] = scaler.transform(X_test_scaled[numeric_features])

    # -------------------------
    # 6) MODEL 1: LINEAR REGRESSION
    # -------------------------
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)

    print("\n================ Linear Regression ================")
    print("R2   :", round(r2_score(y_test, lr_pred), 4))
    print("MAE  :", round(mean_absolute_error(y_test, lr_pred), 2))
    print("RMSE :", round(rmse(y_test, lr_pred), 2))

    # -------------------------
    # 7) MODEL 2: RANDOM FOREST REGRESSOR
    # -------------------------
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)  # RF does not need scaling
    rf_pred = rf.predict(X_test)

    print("\n================ Random Forest Regressor ================")
    print("R2   :", round(r2_score(y_test, rf_pred), 4))
    print("MAE  :", round(mean_absolute_error(y_test, rf_pred), 2))
    print("RMSE :", round(rmse(y_test, rf_pred), 2))

    # -------------------------
    # 8) SAVE BEST MODEL
    # -------------------------
    # Decide best by R2 score
    lr_r2 = r2_score(y_test, lr_pred)
    rf_r2 = r2_score(y_test, rf_pred)

    if rf_r2 >= lr_r2:
        best_model_name = "random_forest"
        best_model = rf
        # RF uses unscaled features
        best_X_test_used = X_test
        best_pred = rf_pred
    else:
        best_model_name = "linear_regression"
        best_model = lr
        # LR uses scaled features
        best_X_test_used = X_test_scaled
        best_pred = lr_pred

    # Save model, scaler, and columns
    joblib.dump(best_model, f"{best_model_name}_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")

    print(f"\n✅ Saved best model: {best_model_name}_model.pkl")
    print("✅ Saved: scaler.pkl")
    print("✅ Saved: feature_columns.pkl")

    # -------------------------
    # 9) SAVE PREDICTIONS CSV
    # -------------------------
    results = best_X_test_used.copy()
    results["actual_price"] = y_test.values
    results["predicted_price"] = best_pred

    results.to_csv("predictions_output.csv", index=False)
    print("✅ Saved predictions: predictions_output.csv")


if __name__ == "__main__":
    main()