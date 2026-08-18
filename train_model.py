import joblib
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def main():
    # Load dataset (no CSV needed)
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    target = "MedHouseVal"
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model (good for deployment, no scaling needed for RF)
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    print("R2   :", round(r2_score(y_test, preds), 4))
    print("MAE  :", round(mean_absolute_error(y_test, preds), 4))
    print("RMSE :", round(rmse(y_test, preds), 4))

    # Save model + feature columns (very important for deployment)
    joblib.dump(model, "model.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")

    print("\n✅ Saved: model.pkl")
    print("✅ Saved: feature_columns.pkl")
    print("\nFeature order:")
    print(list(X.columns))


if __name__ == "__main__":
    main()