from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import pandas as pd

def prepare_features(df):
    cat_cols = ["Education", "LivingWith"]

    ohe = OneHotEncoder(sparse_output=False)
    encoded = ohe.fit_transform(df[cat_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(cat_cols),
        index=df.index
    )

    df = pd.concat([df.drop(columns=cat_cols), encoded_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    return X_scaled, scaler, df


def train_model(X_scaled, n_clusters=4):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_scaled)
    return model, labels