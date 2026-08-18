import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =========================
# 1) LOAD DATA
# =========================
df = pd.read_csv("diamonds.csv")
print("Original shape:", df.shape)

# =========================
# 2) DROP USELESS COLUMN
# =========================
if "Unnamed: 0" in df.columns:
    df = df.drop("Unnamed: 0", axis=1)

# =========================
# 3) BASIC CLEANING
# =========================
# Remove duplicates (safe)
df = df.drop_duplicates()

# Handle missing values (safe even if none)
num_cols = df.select_dtypes(include=["number"]).columns
# FIX: include both object + string to avoid pandas warning
cat_cols = df.select_dtypes(include=["object", "string"]).columns

df[num_cols] = df[num_cols].fillna(df[num_cols].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# =========================
# 4) FIX INVALID VALUES (x/y/z cannot be 0)
# =========================
if set(["x", "y", "z"]).issubset(df.columns):
    df = df[(df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)]

print("After cleaning invalid x/y/z:", df.shape)

# =========================
# 5) OUTLIER HANDLING (OPTIONAL but GOOD)
# =========================
# IQR-based outlier removal (safer: only on key physical size columns)
def remove_outliers_iqr(data, col):
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return data[(data[col] >= low) & (data[col] <= high)]

# Safer outlier removal: apply only to these columns (reduces row loss)
cols_for_outliers = [c for c in ["carat", "x", "y", "z"] if c in df.columns]
for col in cols_for_outliers:
    df = remove_outliers_iqr(df, col)

print("After outlier removal:", df.shape)

# =========================
# 6) ENCODE CATEGORICAL COLUMNS
# =========================
# One-hot encoding is best for cut/color/clarity
df = pd.get_dummies(df, columns=["cut", "color", "clarity"], drop_first=True)

# =========================
# 7) FEATURE SCALING
# =========================
# Scale numeric input features (NOT target 'price')
scaler = StandardScaler()

feature_num_cols = [c for c in ["carat", "depth", "table", "x", "y", "z"] if c in df.columns]
df[feature_num_cols] = scaler.fit_transform(df[feature_num_cols])

# =========================
# 8) SPLIT X and y
# =========================
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train:", X_train.shape, "| X_test:", X_test.shape)

# =========================
# 9) SAVE PROCESSED DATA
# =========================
df.to_csv("diamonds_preprocessed.csv", index=False)
print("Saved: diamonds_preprocessed.csv")

train_df = X_train.copy()
train_df["price"] = y_train.values
test_df = X_test.copy()
test_df["price"] = y_test.values

train_df.to_csv("diamonds_train.csv", index=False)
test_df.to_csv("diamonds_test.csv", index=False)

print("Saved: diamonds_train.csv and diamonds_test.csv")
