from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pandas as pd

def prepare_features(df):

    # using only selected features (App input = Model training features)
    selected_features = ["Income", "TotalSpending", "Age", "TotalChildren"]

    df_final = df[selected_features].copy()

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(df_final)

    return X_scaled, scaler, df_final


def train_model(X_scaled, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    return model, labels