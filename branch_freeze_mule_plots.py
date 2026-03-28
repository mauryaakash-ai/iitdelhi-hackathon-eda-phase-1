import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ACCOUNTS_FILE = "accounts.csv"
LABELS_FILE = "train_labels.csv"
OUTPUT_FILE = "branch_freeze_mule_dotplots.png"


def load_data():
    accounts = pd.read_csv(
        ACCOUNTS_FILE,
        usecols=["account_id", "branch_code", "account_status"],
        dtype={"account_id": "string", "branch_code": "string", "account_status": "category"},
    )
    labels = pd.read_csv(
        LABELS_FILE, usecols=["account_id", "is_mule"], dtype={"account_id": "string", "is_mule": "Int8"}
    )
    return accounts, labels


def aggregate_branch(accounts, labels):
    accounts["is_frozen"] = (accounts["account_status"] == "frozen").astype("int8")

    # All accounts: freeze counts
    branch_freeze = (
        accounts.groupby("branch_code")
        .agg(total_accounts=("account_id", "count"), frozen_accounts=("is_frozen", "sum"))
        .assign(freeze_rate=lambda d: d["frozen_accounts"] / d["total_accounts"])
        .reset_index()
    )

    # Labeled accounts: mule counts (labels only cover a subset)
    labeled = accounts.merge(labels, on="account_id", how="inner")
    branch_mule = (
        labeled.groupby("branch_code")
        .agg(labeled_accounts=("account_id", "count"), mule_accounts=("is_mule", "sum"))
        .assign(mule_rate=lambda d: d["mule_accounts"] / d["labeled_accounts"])
        .reset_index()
    )

    merged = branch_freeze.merge(branch_mule, on="branch_code", how="left")
    num_cols = merged.columns.difference(["branch_code"])
    merged[num_cols] = merged[num_cols].fillna(0)

    merged["freeze_rate"] = merged["freeze_rate"].astype(float)
    merged["mule_rate"] = merged["mule_rate"].astype(float)
    merged["total_accounts"] = merged["total_accounts"].astype(int)
    merged["frozen_accounts"] = merged["frozen_accounts"].astype(int)
    merged["labeled_accounts"] = merged["labeled_accounts"].astype(int)
    merged["mule_accounts"] = merged["mule_accounts"].astype(int)
    return merged


def plot_dotcharts(branch_df):
    top_freeze = branch_df.sort_values("frozen_accounts", ascending=False).head(20)
    top_mule = branch_df.sort_values("mule_accounts", ascending=False).head(20)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=False)

    sns.scatterplot(
        data=top_freeze,
        x="frozen_accounts",
        y="branch_code",
        size="freeze_rate",
        sizes=(50, 200),
        hue="freeze_rate",
        palette="Reds",
        ax=axes[0],
        legend="brief",
    )
    axes[0].set_title("Top branches by frozen accounts")
    axes[0].set_xlabel("Frozen accounts (count)")
    axes[0].set_ylabel("Branch code")

    sns.scatterplot(
        data=top_mule,
        x="mule_accounts",
        y="branch_code",
        size="mule_rate",
        sizes=(50, 200),
        hue="mule_rate",
        palette="Blues",
        ax=axes[1],
        legend="brief",
    )
    axes[1].set_title("Top branches by labeled mules")
    axes[1].set_xlabel("Mule accounts (count)")
    axes[1].set_ylabel("Branch code")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=200)
    plt.close()


def main():
    accounts, labels = load_data()
    branch_df = aggregate_branch(accounts, labels)
    plot_dotcharts(branch_df)
    branch_df.sort_values("frozen_accounts", ascending=False).to_csv(
        "branch_freeze_mule_stats.csv", index=False
    )
    print(f"Saved plots to {OUTPUT_FILE} and stats to branch_freeze_mule_stats.csv")


if __name__ == "__main__":
    main()
