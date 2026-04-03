import pandas as pd

from src.preprocessing import preprocess_data
from src.clustering import prepare_features, train_model
from src.utils import (
    get_cluster_summary,
    plot_cluster_counts,
    plot_income_vs_spending,
    save_object
)

def main():
    # 1. Load data
    df = pd.read_csv("data/smartcart_customers.csv")

    # 2. Preprocess
    df = preprocess_data(df)

    # 3. Feature prep
    X_scaled, scaler, df_final = prepare_features(df)

    # 4. Train model
    model, labels = train_model(X_scaled)

    # 5. Assign clusters
    df_final["Cluster"] = labels

    # 6. Analysis (using utils)
    print("\nCluster Summary:\n")
    print(get_cluster_summary(df_final))

    plot_cluster_counts(df_final)
    plot_income_vs_spending(df_final)

    # 7. Save model
    save_object(model, "models/kmeans_model.pkl")
    save_object(scaler, "models/scaler.pkl")

    print("\n✅ Project pipeline completed successfully!")

if __name__ == "__main__":
    main()