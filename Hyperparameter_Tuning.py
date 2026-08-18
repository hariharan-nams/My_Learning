# ============================================================
# House Price Regression - Hyperparameter Tuning
# Dataset: California Housing (built-in sklearn dataset)
# Model: RandomForestRegressor + RandomizedSearchCV
# File: HousePrice_HyperTuning.py
# ============================================================

import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

import joblib


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    # -------------------------
    # 1) LOAD DATASET (No CSV needed)
    # -------------------------
    data = fetch_california_housing(as_frame=True)
    df = data.frame  # includes features + target

    # Target column name in this dataset: MedHouseVal
    target = "MedHouseVal"

    X = df.drop(columns=[target])
    y = df[target]

    print("Dataset shape:", df.shape)
    print("Features:", list(X.columns))
    print("Target:", target)

    # -------------------------
    # 2) TRAIN/TEST SPLIT
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------
    # 3) PIPELINE (Scaler + Model)
    # -------------------------
    # Random Forest doesn't require scaling, but using scaler in pipeline is OK
    # and helps if you change model later (SVR, Ridge, etc.)
    pipe = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    # -------------------------
    # 4) HYPERPARAMETER SPACE
    # -------------------------
    param_dist = {
        "model__n_estimators": [200, 300, 500, 800],
        "model__max_depth": [None, 10, 20, 30, 40, 60],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", "log2", 0.5, 0.75, None],
        "model__bootstrap": [True, False],
    }

    # -------------------------
    # 5) RANDOMIZED SEARCH CV
    # -------------------------
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=25,            # increase to 50 for better results (slower)
        scoring="r2",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("\n✅ Best Parameters:")
    print(search.best_params_)
    print("✅ Best CV R2:", round(search.best_score_, 4))

    # -------------------------
    # 6) TEST EVALUATION
    # -------------------------
    preds = best_model.predict(X_test)

    test_r2 = r2_score(y_test, preds)
    test_mae = mean_absolute_error(y_test, preds)
    test_rmse = rmse(y_test, preds)

    print("\n================ TEST RESULTS (TUNED MODEL) ================")
    print("Test R2   :", round(test_r2, 4))
    print("Test MAE  :", round(test_mae, 4))
    print("Test RMSE :", round(test_rmse, 4))

    # -------------------------
    # 7) SAVE MODEL + PREDICTIONS
    # -------------------------
    joblib.dump(best_model, "houseprice_tuned_model.pkl")
    print("\n✅ Saved model: houseprice_tuned_model.pkl")

    out = X_test.copy()
    out["actual"] = y_test.values
    out["predicted"] = preds
    out.to_csv("houseprice_tuned_predictions.csv", index=False)
    print("✅ Saved predictions: houseprice_tuned_predictions.csv")


if __name__ == "__main__":
    main()