# ============================================================
# Unsupervised Learning - Customer Segmentation
# File: Unsupervised_Learning.py
# Dataset: Mall_Customers.csv
# Algorithm: K-Means Clustering
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def main():

    # -------------------------
    # 1) LOAD DATA
    # -------------------------
    df = pd.read_csv("Mall_Customers.csv")

    print("Dataset Shape:", df.shape)
    print(df.head())

    # -------------------------
    # 2) DATA PREPROCESSING
    # -------------------------
    # Drop CustomerID
    df = df.drop("CustomerID", axis=1)

    # Encode Gender
    df["Gender"] = df["Gender"].map({"Male": 0, "Female": 1})

    # -------------------------
    # 3) FEATURE SELECTION
    # -------------------------
    # We use Income & Spending Score for clustering
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------------
    # 4) ELBOW METHOD
    # -------------------------
    wcss = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)

    plt.figure()
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()

    # -------------------------
    # 5) KMEANS (Optimal k = 5 usually)
    # -------------------------
    kmeans = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df["Cluster"] = clusters

    # -------------------------
    # 6) VISUALIZE CLUSTERS
    # -------------------------
    plt.figure()
    sns.scatterplot(
        x=df["Annual Income (k$)"],
        y=df["Spending Score (1-100)"],
        hue=df["Cluster"],
        palette="Set1"
    )
    plt.title("Customer Segments")
    plt.show()

    # -------------------------
    # 7) SILHOUETTE SCORE
    # -------------------------
    score = silhouette_score(X_scaled, clusters)
    print("Silhouette Score:", round(score, 3))

    print("\n✅ Unsupervised Learning Completed!")


if __name__ == "__main__":
    main()