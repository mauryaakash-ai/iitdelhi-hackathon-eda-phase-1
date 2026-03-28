import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


TRANSACTION_PARTS = [f"transactions_part_{i}.csv" for i in range(6)]
LABEL_FILE = "train_labels.csv"
ACCOUNTS_FILE = "accounts.csv"
OUTPUT_DIR = Path(".")


def load_accounts():
    """Lightweight account-level features for clustering."""
    accounts = pd.read_csv(
        ACCOUNTS_FILE,
        dtype={
            "account_id": "string",
            "account_status": "category",
            "product_code": "category",
            "product_family": "category",
            "currency_code": "Int16",
            "branch_code": "category",
            "branch_pin": "Int32",
            "avg_balance": "float32",
            "monthly_avg_balance": "float32",
            "quarterly_avg_balance": "float32",
            "daily_avg_balance": "float32",
            "nomination_flag": "category",
            "cheque_allowed": "category",
            "cheque_availed": "category",
            "kyc_compliant": "category",
            "rural_branch": "category",
        },
        parse_dates=[
            "account_opening_date",
            "last_mobile_update_date",
            "last_kyc_date",
            "freeze_date",
            "unfreeze_date",
        ],
    )

    flag_cols = [
        "nomination_flag",
        "cheque_allowed",
        "cheque_availed",
        "kyc_compliant",
        "rural_branch",
    ]
    for col in flag_cols:
        if col in accounts.columns:
            accounts[col] = accounts[col].map({"Y": 1, "N": 0})

    accounts["account_age_days"] = (
        pd.Timestamp("today").normalize() - accounts["account_opening_date"]
    ).dt.days

    keep_cols = [
        "account_id",
        "product_family",
        "account_status",
        "avg_balance",
        "monthly_avg_balance",
        "daily_avg_balance",
        "nomination_flag",
        "cheque_allowed",
        "cheque_availed",
        "kyc_compliant",
        "rural_branch",
        "account_age_days",
    ]

    return accounts[keep_cols]


def aggregate_transactions():
    """Aggregate transaction-level signals per account without loading everything at once."""
    part_aggs = []

    txn_dtypes = {
        "transaction_id": "string",
        "account_id": "string",
        "transaction_timestamp": "string",
        "mcc_code": "Int32",
        "channel": "category",
        "amount": "float32",
        "txn_type": "category",
        "counterparty_id": "string",
    }

    for path in TRANSACTION_PARTS:
        df = pd.read_csv(path, dtype=txn_dtypes, parse_dates=["transaction_timestamp"])

        df["is_credit"] = (df["txn_type"] == "C").astype("int8")
        df["is_debit"] = (df["txn_type"] == "D").astype("int8")
        df["credit_amount"] = np.where(df["txn_type"] == "C", df["amount"], 0.0)
        df["debit_amount"] = np.where(df["txn_type"] == "D", df["amount"], 0.0)
        df["amount_sq"] = df["amount"] ** 2

        df["is_upi"] = df["channel"].isin(["UPC", "UPD"]).astype("int8")
        df["is_imps"] = (df["channel"] == "IPM").astype("int8")
        df["is_neft"] = (df["channel"] == "NTD").astype("int8")
        df["is_atm"] = (df["channel"] == "ATW").astype("int8")
        df["is_cash"] = df["channel"].isin(["CSD", "OCD"]).astype("int8")

        part_aggs.append(
            df.groupby("account_id").agg(
                txn_count=("transaction_id", "count"),
                credit_count=("is_credit", "sum"),
                debit_count=("is_debit", "sum"),
                credit_amount=("credit_amount", "sum"),
                debit_amount=("debit_amount", "sum"),
                amount_sum=("amount", "sum"),
                amount_sq_sum=("amount_sq", "sum"),
                upi_count=("is_upi", "sum"),
                imps_count=("is_imps", "sum"),
                neft_count=("is_neft", "sum"),
                atm_count=("is_atm", "sum"),
                cash_count=("is_cash", "sum"),
                first_txn=("transaction_timestamp", "min"),
                last_txn=("transaction_timestamp", "max"),
            ).reset_index()
        )

        del df

    agg = pd.concat(part_aggs, ignore_index=True)

    agg = (
        agg.groupby("account_id")
        .agg(
            txn_count=("txn_count", "sum"),
            credit_count=("credit_count", "sum"),
            debit_count=("debit_count", "sum"),
            credit_amount=("credit_amount", "sum"),
            debit_amount=("debit_amount", "sum"),
            amount_sum=("amount_sum", "sum"),
            amount_sq_sum=("amount_sq_sum", "sum"),
            upi_count=("upi_count", "sum"),
            imps_count=("imps_count", "sum"),
            neft_count=("neft_count", "sum"),
            atm_count=("atm_count", "sum"),
            cash_count=("cash_count", "sum"),
            first_txn=("first_txn", "min"),
            last_txn=("last_txn", "max"),
        )
        .reset_index()
    )

    agg["net_amount"] = agg["credit_amount"] - agg["debit_amount"]
    agg["amount_mean"] = agg["amount_sum"] / agg["txn_count"].clip(lower=1)
    agg["amount_std"] = np.sqrt(
        np.maximum(
            (agg["amount_sq_sum"] / agg["txn_count"].clip(lower=1))
            - agg["amount_mean"] ** 2,
            0,
        )
    )

    active_days = (agg["last_txn"] - agg["first_txn"]).dt.days.clip(lower=1)
    agg["txns_per_day"] = agg["txn_count"] / active_days

    agg["credit_debit_ratio"] = agg["credit_count"] / agg["debit_count"].replace(
        0, np.nan
    )
    agg["credit_debit_ratio"].fillna(agg["credit_count"], inplace=True)

    agg["amount_ratio"] = agg["credit_amount"] / agg["debit_amount"].replace(
        0, np.nan
    )
    agg["amount_ratio"].fillna(agg["credit_amount"], inplace=True)

    for channel in ["upi", "imps", "neft", "atm", "cash"]:
        agg[f"{channel}_ratio"] = agg[f"{channel}_count"] / agg["txn_count"]

    return agg


def build_feature_matrix():
    txn_features = aggregate_transactions()
    account_features = load_accounts()

    features = txn_features.merge(account_features, on="account_id", how="left")

    # One-hot encode low-cardinality categoricals
    features = pd.get_dummies(
        features, columns=["product_family", "account_status"], dummy_na=True
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    feature_cols = [
        col
        for col in features.columns
        if col
        not in [
            "account_id",
            "first_txn",
            "last_txn",
        ]
    ]

    X = features[feature_cols].astype("float32")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return features, X_scaled


def fit_clusters(X_scaled, n_clusters=6, random_state=42):
    model = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=random_state, batch_size=2048, max_iter=200
    )
    cluster_labels = model.fit_predict(X_scaled)
    return model, cluster_labels


def summarize_clusters(features, cluster_labels):
    labels_df = pd.read_csv(
        LABEL_FILE, dtype={"account_id": "string", "is_mule": "Int8"}
    )
    features["cluster"] = cluster_labels

    merged = features.merge(labels_df, on="account_id", how="left")

    summary = (
        merged.groupby("cluster")
        .agg(
            total_accounts=("account_id", "count"),
            labeled_accounts=("is_mule", "count"),
            mule_rate=("is_mule", "mean"),
            median_txns_per_day=("txns_per_day", "median"),
            median_amount_mean=("amount_mean", "median"),
            median_amount_std=("amount_std", "median"),
        )
        .reset_index()
    )

    summary.sort_values("mule_rate", ascending=False, inplace=True)
    summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    print("Cluster summary saved to cluster_summary.csv")
    print(summary)

    return merged, summary


def plot_clusters(features, summary, X_scaled):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    features["pca1"] = coords[:, 0]
    features["pca2"] = coords[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=features,
        x="pca1",
        y="pca2",
        hue="cluster",
        palette="tab10",
        alpha=0.6,
        s=20,
        linewidth=0,
    )
    plt.title("Account Clusters (PCA projection)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_scatter.png", dpi=200)
    plt.close()
    print("Cluster scatter plot saved to cluster_scatter.png")

    plt.figure(figsize=(8, 4))
    sns.barplot(data=summary, x="cluster", y="mule_rate", palette="Reds")
    plt.title("Mule rate per cluster (train labels only)")
    plt.ylabel("Mule rate")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_mule_rate.png", dpi=200)
    plt.close()
    print("Mule-rate bar plot saved to cluster_mule_rate.png")


def main():
    print("Building feature matrix...")
    features, X_scaled = build_feature_matrix()

    print("Fitting MiniBatchKMeans clusters...")
    model, cluster_labels = fit_clusters(X_scaled, n_clusters=6, random_state=42)

    print("Summarizing clusters against train labels...")
    merged, summary = summarize_clusters(features, cluster_labels)

    print("Plotting clusters...")
    plot_clusters(merged, summary, X_scaled)

    merged[["account_id", "cluster"]].to_csv(
        OUTPUT_DIR / "account_cluster_assignments.csv", index=False
    )
    print("Cluster assignments saved to account_cluster_assignments.csv")


if __name__ == "__main__":
    main()
