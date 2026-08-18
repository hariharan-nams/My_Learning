# ============================================================
# Diamonds Data Visualization Assignment
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():

    # -------------------------
    # 1) LOAD DATA
    # -------------------------
    df = pd.read_csv("diamonds.csv")

    # Drop unnecessary column
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    print("Shape:", df.shape)
    print(df.head())

    SAVE = True   # Make sure this is aligned inside main()

    # -------------------------
    # 2) HISTOGRAM: Price Distribution
    # -------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price"], bins=50)
    plt.title("Distribution of Diamond Prices")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    if SAVE:
        plt.savefig("01_price_distribution.png", dpi=200, bbox_inches="tight")
    plt.show()

    # -------------------------
    # 3) COUNTPLOT: Diamonds by Cut
    # -------------------------
    plt.figure(figsize=(8, 5))
    sns.countplot(x="cut", data=df)
    plt.title("Count of Diamonds by Cut")
    plt.xticks(rotation=45)
    if SAVE:
        plt.savefig("02_count_by_cut.png", dpi=200, bbox_inches="tight")
    plt.show()

    # -------------------------
    # 4) BARPLOT: Average Price by Cut
    # -------------------------
    plt.figure(figsize=(8, 5))
    sns.barplot(x="cut", y="price", data=df)
    plt.title("Average Price by Cut")
    plt.xticks(rotation=45)
    if SAVE:
        plt.savefig("03_avg_price_by_cut.png", dpi=200, bbox_inches="tight")
    plt.show()

    # -------------------------
    # 5) SCATTER: Carat vs Price
    # -------------------------
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="carat", y="price", data=df, s=20)
    plt.title("Carat vs Price")
    if SAVE:
        plt.savefig("04_carat_vs_price.png", dpi=200, bbox_inches="tight")
    plt.show()

    # -------------------------
    # 6) HEATMAP
    # -------------------------
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True)
    plt.title("Correlation Heatmap")
    if SAVE:
        plt.savefig("05_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.show()

    # -------------------------
    # 7) BOXPLOT
    # -------------------------
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df["price"])
    plt.title("Boxplot of Price")
    if SAVE:
        plt.savefig("06_price_boxplot.png", dpi=200, bbox_inches="tight")
    plt.show()

    print("✅ Visualization Completed!")


if __name__ == "__main__":
    main()