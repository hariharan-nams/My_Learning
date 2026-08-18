
import os
import warnings
warnings.filterwarnings("ignore")

# ---- Windows-safe plotting (no Tkinter popups) ----
import matplotlib
matplotlib.use("Agg")  # prevents Tkinter GUI cleanup errors

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

sns.set_style("whitegrid")


def eval_metrics(y_true, y_pred):
    """Print classification report, return metrics dict + confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1_val = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(classification_report(y_true, y_pred, zero_division=0))

    metrics = {
        "accuracy_score": acc,
        "precision_score": precision,
        "recall_score": recall,
        "f1_score": f1_val,
        "f1_macro": f1_macro
    }
    return metrics, cm


def run_grid_search(model, param_grid, x_train, y_train, scoring="f1"):
    """GridSearchCV with 5-fold KFold."""
    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=kf,
        scoring=scoring,
        n_jobs=-1
    )
    gs.fit(x_train, y_train)

    print(f"Best Parameters: {gs.best_params_}")
    print(f"Best CV {scoring.upper()} Score: {gs.best_score_:.4f}")

    feat_imp = getattr(gs.best_estimator_, "feature_importances_", None)
    print(f"Feature Importances: {feat_imp if feat_imp is not None else 'N/A'}")

    return gs.best_estimator_


def save_confusion_matrix(cm, title, save_path):
    """Save confusion matrix as PNG."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_decision_tree(model, feature_names, save_path):
    """Save decision tree plot as PNG."""
    plt.figure(figsize=(14, 9), dpi=300, edgecolor="k")
    plot_tree(
        model,
        filled=True,
        fontsize=6,
        feature_names=feature_names,
        class_names=["Rejected", "Approved"]
    )
    plt.title("Best Decision Tree")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():

    DATA_PATH = r"E:\Study Materials\Uptor\DS_Uptorbatch_113\Assignment\Final_Project\loan_approval_dataset.csv"

    # Output folder for plots
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load dataset
    df = pd.read_csv(DATA_PATH)

    # 2) Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # 3) Drop id column if exists
    if "loan_id" in df.columns:
        df = df.drop("loan_id", axis=1)

    # 4) Strip spaces in key categorical columns
    for col in ["education", "self_employed", "loan_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 5) Encode categorical -> numeric
    if "education" in df.columns:
        df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
    if "self_employed" in df.columns:
        df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
    if "loan_status" in df.columns:
        df["loan_status"] = df["loan_status"].map({"Approved": 1, "Rejected": 0})

    # Convert mapped columns to int
    for col in ["education", "self_employed", "loan_status"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # 6) Feature lists
    num_f = [
        "income_annum", "loan_amount", "loan_term", "cibil_score",
        "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value"
    ]

    # 7) Scale numerical features
    scaler = StandardScaler()
    existing_num = [c for c in num_f if c in df.columns]
    df[existing_num] = scaler.fit_transform(df[existing_num])

    # 8) Split X / y
    if "loan_status" not in df.columns:
        raise ValueError("Target column 'loan_status' not found.")

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=7, stratify=y
    )

    # ============================================================
    # LOGISTIC REGRESSION (BASELINE)
    # ============================================================
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION (BASELINE)")
    print("=" * 60)

    log_reg = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        random_state=7
    )
    log_reg.fit(x_train, y_train)

    y_pred = log_reg.predict(x_test)
    metrics, cm = eval_metrics(y_test, y_pred)
    print("Metrics:", metrics)

    save_confusion_matrix(
        cm,
        "Confusion Matrix - Logistic Regression (Baseline)",
        os.path.join(OUTPUT_DIR, "cm_logreg_baseline.png")
    )

    # ============================================================
    # LOGISTIC REGRESSION (GRID SEARCH)
    # ============================================================
    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION (GRID SEARCH)")
    print("=" * 60)

    # Safer grid to avoid invalid solver/penalty combos
    log_param_grid = [
        {"penalty": ["l2"], "C": [0.001, 0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]},
        {"penalty": ["l1"], "C": [0.001, 0.01, 0.1, 1, 10, 100], "solver": ["liblinear"]},
        {"penalty": ["elasticnet"], "C": [0.001, 0.01, 0.1, 1, 10], "solver": ["saga"], "l1_ratio": [0, 0.5, 1]},
    ]

    best_log = run_grid_search(
        LogisticRegression(max_iter=5000, random_state=7),
        log_param_grid,
        x_train,
        y_train,
        scoring="f1"
    )

    y_pred = best_log.predict(x_test)
    metrics, cm = eval_metrics(y_test, y_pred)
    print("Metrics:", metrics)

    save_confusion_matrix(
        cm,
        "Confusion Matrix - Logistic Regression (Best)",
        os.path.join(OUTPUT_DIR, "cm_logreg_best.png")
    )

    # ============================================================
    # DECISION TREE (BASELINE)
    # ============================================================
    print("\n" + "=" * 60)
    print("DECISION TREE (BASELINE)")
    print("=" * 60)

    dt = DecisionTreeClassifier(criterion="gini", random_state=7)
    dt.fit(x_train, y_train)

    y_pred = dt.predict(x_test)
    metrics, cm = eval_metrics(y_test, y_pred)
    print("Metrics:", metrics)

    save_confusion_matrix(
        cm,
        "Confusion Matrix - Decision Tree (Baseline)",
        os.path.join(OUTPUT_DIR, "cm_dt_baseline.png")
    )

    # ============================================================
    # DECISION TREE (GRID SEARCH)
    # ============================================================
    print("\n" + "=" * 60)
    print("DECISION TREE (GRID SEARCH)")
    print("=" * 60)

    dt_param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
        "min_samples_split": [2, 3, 4, 6, 8, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5],
        "max_leaf_nodes": [None, 10, 20, 30, 40, 50],
    }

    best_dt = run_grid_search(
        DecisionTreeClassifier(random_state=7),
        dt_param_grid,
        x_train,
        y_train,
        scoring="f1"
    )

    y_pred = best_dt.predict(x_test)
    metrics, cm = eval_metrics(y_test, y_pred)
    print("Metrics:", metrics)

    # plot confusion matrix for best_dt (not best_log)
    save_confusion_matrix(
        cm,
        "Confusion Matrix - Decision Tree (Best)",
        os.path.join(OUTPUT_DIR, "cm_dt_best.png")
    )

    # Save best decision tree plot
    save_decision_tree(
        best_dt,
        feature_names=list(X.columns),
        save_path=os.path.join(OUTPUT_DIR, "best_decision_tree.png")
    )

    print("\nSaved output images to:", os.path.abspath(OUTPUT_DIR))
    print(" - cm_logreg_baseline.png")
    print(" - cm_logreg_best.png")
    print(" - cm_dt_baseline.png")
    print(" - cm_dt_best.png")
    print(" - best_decision_tree.png")


if __name__ == "__main__":
    main()