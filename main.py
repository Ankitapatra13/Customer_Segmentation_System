import pandas as pd
import pickle
import os

from src.preprocessing import preprocess_data
from src.clustering import prepare_features, train_model
from src.utils import (
    get_cluster_summary,
    plot_cluster_counts,
    plot_income_vs_spending,
    save_object
)


def main():
    model_path = "models/kmeans_model.pkl"
    scaler_path = "models/scaler.pkl"
    features_path = "models/all_features.pkl"
    ## adding cluster summarys into models folder 
    cluster_path = "models/cluster_summary.csv"
    final_data_path = "models/preprocessed_data.csv"

    # Create models folder if not exists
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(model_path):
        print("🚀 Training model...")

        # 1. Load data
        df = pd.read_csv("data/smartcart_customers.csv")

        # 2. Preprocess
        df = preprocess_data(df)

        # 3. Feature preparation
        X_scaled, scaler, df_final = prepare_features(df)

        # 4. Save feature columns (for app)
        with open(features_path, "wb") as f:
            pickle.dump(df_final.columns.tolist(), f)

        # 5. Train model
        model, labels = train_model(X_scaled)

        # 6. Assign clusters
        df_final["Cluster"] = labels

        # 7. Cluster summary
        print("\n📊 Cluster Summary:\n")
        summary = get_cluster_summary(df_final)
        print(summary)

        # 8. Visualizations
        plot_cluster_counts(df_final)
        plot_income_vs_spending(df_final)

        # 9. Save model and scaler
        save_object(model, model_path)
        save_object(scaler, scaler_path)
        summary.to_csv(cluster_path,index=True)
        df_final.to_csv(final_data_path,index=False)

        print("\n✅ Project pipeline completed successfully!")

    else:
        print("⚠️ Model already exists. Skipping training.")


if __name__ == "__main__":
    main()