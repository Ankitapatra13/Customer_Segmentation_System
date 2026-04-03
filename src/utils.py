import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def get_cluster_summary(df):
    return df.groupby("Cluster").mean(numeric_only=True)

def plot_cluster_counts(df):
    sns.countplot(x="Cluster", data=df)
    plt.title("Cluster Distribution")
    plt.show()

def plot_income_vs_spending(df):
    sns.scatterplot(
        x=df["Income"],
        y=df["TotalSpending"],
        hue=df["Cluster"]
    )
    plt.title("Income vs Spending")
    plt.show()

def save_object(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_object(path):
    with open(path, "rb") as f:
        return pickle.load(f)