# ============================================================
# Feature Selection Assignment
# Dataset: California Housing (Built-in sklearn)
# ============================================================

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def main():

    # -------------------------
    # 1) LOAD DATA
    # -------------------------
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    print("Dataset shape:", df.shape)
    print(df.head())

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # -------------------------
    # 2) CORRELATION METHOD
    # -------------------------
    print("\n=== Correlation with Target ===")
    corr = df.corr(numeric_only=True)["MedHouseVal"].sort_values(ascending=False)
    print(corr)

    # -------------------------
    # 3) SELECTKBEST METHOD
    # -------------------------
    print("\n=== SelectKBest (Top 5 Features) ===")

    selector = SelectKBest(score_func=f_regression, k=5)
    selector.fit(X, y)

    selected_features = X.columns[selector.get_support()]
    print("Selected Features:", list(selected_features))

    # -------------------------
    # 4) RANDOM FOREST FEATURE IMPORTANCE
    # -------------------------
    print("\n=== Random Forest Feature Importance ===")

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X, y)

    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    print(importances)

    # -------------------------
    # 5) TOP 5 IMPORTANT FEATURES
    # -------------------------
    top_features = importances.head(5)
    print("\nTop 5 Important Features:")
    print(top_features)

    print("\n✅ Feature Selection Completed!")


if __name__ == "__main__":
    main()