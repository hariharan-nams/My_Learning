# ============================================================
# Statistics + Advanced Statistics
# Dataset: California Housing (Built-in sklearn)
#
# Includes:
# BASIC:
# - Mean, Median, Mode, Variance, Std Dev
# - Distribution plot
# - Z-score
# - Correlation matrix
# - T-test
#
# ADVANCED:
# - ANOVA
# - Chi-square test
# - Confidence interval
# - Normal distribution plotting (KDE)
# - Central Limit Theorem simulation
# ============================================================

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error


# -------------------------
# BASIC STATISTICS
# -------------------------
def basic_statistics(df: pd.DataFrame, column: str = "MedInc") -> None:
    print("\n================ Basic Statistics ================")

    mean = df[column].mean()
    median = df[column].median()
    mode = df[column].mode()[0]
    variance = df[column].var()
    std_dev = df[column].std()

    print("Column:", column)
    print("Mean:", round(float(mean), 3))
    print("Median:", round(float(median), 3))
    print("Mode:", round(float(mode), 3))
    print("Variance:", round(float(variance), 3))
    print("Standard Deviation:", round(float(std_dev), 3))

    # Distribution Plot
    plt.figure(figsize=(8, 4))
    df[column].hist(bins=40)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

    # Z-score
    print("\n================ Z-Score Example ================")
    sample_value = df[column].iloc[0]
    z_score = (sample_value - mean) / std_dev
    print("Sample Value:", round(float(sample_value), 3))
    print("Z-Score:", round(float(z_score), 3))

    # Correlation
    print("\n================ Correlation Matrix ================")
    corr_matrix = df.corr(numeric_only=True)
    print(corr_matrix)

    # T-test (High income vs Low income) on house value
    print("\n================ Hypothesis Testing (T-Test) ================")
    high_income = df[df["MedInc"] > df["MedInc"].median()]["MedHouseVal"]
    low_income = df[df["MedInc"] <= df["MedInc"].median()]["MedHouseVal"]

    t_stat, p_value = stats.ttest_ind(high_income, low_income)
    print("T-Statistic:", round(float(t_stat), 3))
    print("P-Value:", float(p_value))

    if p_value < 0.05:
        print("Result: Reject H0 (Significant difference)")
    else:
        print("Result: Fail to Reject H0")


# -------------------------
# ADVANCED STATISTICS
# -------------------------
def anova_test(df: pd.DataFrame) -> None:
    print("\n================ ANOVA (One-way) ================")

    df2 = df.copy()
    df2["Income_Group"] = pd.qcut(df2["MedInc"], 3, labels=["Low", "Medium", "High"])

    group_low = df2[df2["Income_Group"] == "Low"]["MedHouseVal"]
    group_med = df2[df2["Income_Group"] == "Medium"]["MedHouseVal"]
    group_high = df2[df2["Income_Group"] == "High"]["MedHouseVal"]

    f_stat, p_value = stats.f_oneway(group_low, group_med, group_high)

    print("F-Statistic:", round(float(f_stat), 4))
    print("P-Value    :", float(p_value))

    if p_value < 0.05:
        print("Result     : Reject H0 (Means are significantly different)")
    else:
        print("Result     : Fail to Reject H0 (No significant difference)")


def chi_square_test(df: pd.DataFrame) -> None:
    print("\n================ Chi-Square Test ================")

    df2 = df.copy()
    df2["Income_Group"] = pd.qcut(df2["MedInc"], 3, labels=["Low", "Medium", "High"])
    df2["Value_Category"] = np.where(
        df2["MedHouseVal"] > df2["MedHouseVal"].median(), "High", "Low"
    )

    contingency = pd.crosstab(df2["Income_Group"], df2["Value_Category"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    print("Contingency Table:\n", contingency)
    print("\nChi2 Statistic:", round(float(chi2), 4))
    print("Degrees of freedom:", int(dof))
    print("P-Value:", float(p))

    if p < 0.05:
        print("Result: Reject H0 (Variables are dependent)")
    else:
        print("Result: Fail to Reject H0 (Variables are independent)")


def confidence_interval(series: pd.Series, confidence: float = 0.95) -> None:
    print("\n================ Confidence Interval ================")

    mean = series.mean()
    std = series.std(ddof=1)
    n = len(series)

    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha / 2)
    margin_error = z_critical * (std / np.sqrt(n))

    lower = mean - margin_error
    upper = mean + margin_error

    print("Column:", series.name)
    print("Mean  :", round(float(mean), 4))
    print(f"{int(confidence*100)}% CI:", (round(float(lower), 4), round(float(upper), 4)))


def normal_distribution_plot(series: pd.Series) -> None:
    print("\n================ Normal Distribution Plot (KDE) ================")

    plt.figure(figsize=(9, 4))
    sns.histplot(series, kde=True, bins=40)
    plt.title(f"Histogram + KDE - {series.name}")
    plt.xlabel(series.name)
    plt.ylabel("Frequency")
    plt.show()


def clt_simulation(series: pd.Series, sample_size: int = 30, n_samples: int = 1000) -> None:
    print("\n================ Central Limit Theorem (CLT) Simulation ================")

    values = series.values
    sample_means = []

    for _ in range(n_samples):
        sample = np.random.choice(values, size=sample_size, replace=True)
        sample_means.append(sample.mean())

    plt.figure(figsize=(9, 4))
    sns.histplot(sample_means, kde=True, bins=40)
    plt.title(f"CLT: Sample Means (size={sample_size}, repeats={n_samples})")
    plt.xlabel("Sample Mean")
    plt.ylabel("Frequency")
    plt.show()

    print("CLT Summary:")
    print("Mean of sample means:", round(float(np.mean(sample_means)), 4))
    print("Std of sample means :", round(float(np.std(sample_means, ddof=1)), 4))


# -------------------------
# MAIN (ONLY ONE)
# -------------------------
def main():
    data = fetch_california_housing(as_frame=True)
    df = data.frame

    print("✅ Dataset Loaded:", df.shape)
    print(df.head())

    # BASIC
    basic_statistics(df, column="MedInc")

    # ADVANCED
    anova_test(df)
    chi_square_test(df)
    confidence_interval(df["MedInc"], confidence=0.95)
    normal_distribution_plot(df["MedInc"])
    clt_simulation(df["MedInc"], sample_size=30, n_samples=1000)

    print("\n✅ Completed: Statistics + Advanced Statistics Assignment")


if __name__ == "__main__":
    main()