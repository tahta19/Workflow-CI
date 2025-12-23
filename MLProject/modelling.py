import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# contoh data dummy (ganti dengan datasetmu)
X = np.random.rand(100, 4)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_clusters = 3

with mlflow.start_run():
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X_scaled)

    score = silhouette_score(X_scaled, model.labels_)

    mlflow.log_param("n_clusters", n_clusters)
    mlflow.log_metric("silhouette_score", score)

    mlflow.sklearn.log_model(model, "model")
