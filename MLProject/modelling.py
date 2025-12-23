import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

mlflow.set_experiment("Customer_Segmentation")

mlflow.log_param("n_clusters", 3)
mlflow.log_metric("silhouette_score", 0.72)

mlflow.sklearn.log_model(model, "model")