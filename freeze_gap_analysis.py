import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


ACCOUNTS_FILE = "accounts.csv"
LABELS_FILE = "train_labels.csv"
OUTPUT_DIR = Path(".")


def load_accounts():
    df = pd.read_csv(
        ACCOUNTS_FILE,
        usecols=[
            "account_id",
            "branch_code",
            "account_opening_date",
            "freeze_date",
            "account_status",
        ],
        dtype={
            "account_id": "string",
            "branch_code": "string",
            "account_status": "category",
        },
    )
    df["account_opening_date"] = pd.to_datetime(
        df["account_opening_date"], errors="coerce"
    )
    df["freeze_date"] = pd.to_datetime(df["freeze_date"], errors="coerce")
    return df


def compute_freeze_to_next_open(accounts: pd.DataFrame) -> pd.DataFrame:
    # Keep only rows with a freeze date
    frozen = accounts.dropna(subset=["freeze_date"]).copy()
    results = []

    for branch, df_branch in accounts.groupby("branch_code"):
        open_dates = (
            pd.to_datetime(df_branch["account_opening_date"], errors="coerce")
            .dropna()
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )
        if len(open_dates) == 0:
            continue

        open_ord = open_dates.view("int64")

        branch_frozen = frozen[frozen["branch_code"] == branch]
        if branch_frozen.empty:
            continue

        for _, row in branch_frozen.iterrows():
            freeze_dt = row["freeze_date"]
            if pd.isna(freeze_dt):
                continue
            freeze_ord = np.datetime64(freeze_dt, "ns").view("int64")
            idx = np.searchsorted(open_ord, freeze_ord, side="right")
            if idx >= len(open_dates):
                continue  # no later opening in this branch
            next_open = open_dates[idx]
            gap_days = (next_open - freeze_dt).days
            results.append(
                {
                    "account_id": row["account_id"],
                    "branch_code": branch,
                    "freeze_date": freeze_dt,
                    "next_open_date": pd.Timestamp(next_open),
                    "gap_days": gap_days,
                }
            )

    gap_df = pd.DataFrame(results)
    return gap_df


def aggregate_branch_gap(
    gap_df: pd.DataFrame, accounts: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    branch_gaps = (
        gap_df.groupby("branch_code")
        .agg(
            samples=("gap_days", "count"),
            median_gap=("gap_days", "median"),
            mean_gap=("gap_days", "mean"),
            p90_gap=("gap_days", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )

    freeze_counts = (
        accounts.dropna(subset=["freeze_date"])
        .groupby("branch_code")
        .size()
        .rename("freeze_events")
        .reset_index()
    )

    branch_mule = (
        accounts.merge(labels, on="account_id", how="inner")
        .groupby("branch_code")
        .agg(
            labeled_accounts=("account_id", "count"),
            mule_accounts=("is_mule", "sum"),
        )
        .assign(
            mule_rate=lambda d: d["mule_accounts"]
            / d["labeled_accounts"].replace(0, np.nan)
        )
        .reset_index()
    )

    branch_gaps = (
        branch_gaps.merge(freeze_counts, on="branch_code", how="left")
        .merge(branch_mule, on="branch_code", how="left")
        .fillna({"mule_rate": 0, "mule_accounts": 0, "labeled_accounts": 0})
    )

    numeric_cols = [
        "samples",
        "median_gap",
        "mean_gap",
        "p90_gap",
        "freeze_events",
        "labeled_accounts",
        "mule_accounts",
        "mule_rate",
    ]
    for col in numeric_cols:
        branch_gaps[col] = branch_gaps[col].astype(float)

    return branch_gaps.sort_values("freeze_events", ascending=False)


def merge_alert_reasons(accounts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    frozen = accounts.dropna(subset=["freeze_date"])
    return frozen.merge(labels, on="account_id", how="left")


def plot_branch_gap(branch_gaps: pd.DataFrame, gap_df: pd.DataFrame):
    top = branch_gaps.head(25)

    plt.figure(figsize=(12, 7))
    scatter = sns.scatterplot(
        data=top,
        x="median_gap",
        y="branch_code",
        size="freeze_events",
        hue="mule_rate",
        palette="coolwarm",
        sizes=(60, 400),
        legend="brief",
    )
    plt.xlabel("Median days: freeze → next account opening (branch)")
    plt.ylabel("Branch code")
    plt.title("Branches with most freeze events\nSize = #freeze events, Color = mule rate (labeled accounts)")
    handles, labels = scatter.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title="Mule rate / Freeze events", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "freeze_gap_branch_bubble.png", dpi=250)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=gap_df,
        x="gap_days",
        hue="label_group",
        bins=40,
        multiple="stack",
        palette={"Mule": "#d73027", "Legit": "#1a9850", "Unlabeled": "#4575b4"},
    )
    plt.xlabel("Days from freeze to next account opening (same branch)")
    plt.ylabel("Number of freeze events")
    plt.title("Freeze→next opening gap by account label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "freeze_gap_hist_by_label.png", dpi=250)
    plt.close()


def load_labels():
    return pd.read_csv(
        LABELS_FILE,
        usecols=["account_id", "is_mule", "alert_reason"],
        dtype={"account_id": "string", "is_mule": "Int8", "alert_reason": "string"},
    )


def main():
    accounts = load_accounts()
    labels = load_labels()

    gap_df = compute_freeze_to_next_open(accounts)
    gap_df = gap_df.merge(labels[["account_id", "is_mule"]], on="account_id", how="left")
    gap_df["label_group"] = gap_df["is_mule"].map({1: "Mule", 0: "Legit"}).fillna("Unlabeled")

    branch_gaps = aggregate_branch_gap(gap_df, accounts, labels)
    plot_branch_gap(branch_gaps, gap_df)

    branch_gaps.to_csv(OUTPUT_DIR / "freeze_to_open_branch_stats.csv", index=False)
    gap_df.to_csv(OUTPUT_DIR / "freeze_to_open_events.csv", index=False)

    alert_df = merge_alert_reasons(accounts, labels)
    alert_counts = (
        alert_df["alert_reason"].value_counts(dropna=True).head(10).reset_index()
    )
    alert_counts.columns = ["alert_reason", "count"]
    alert_counts.to_csv(OUTPUT_DIR / "frozen_accounts_alert_reasons.csv", index=False)

    print("Saved gap plots and stats:")
    print(" - freeze_gap_branch_bubble.png")
    print(" - freeze_gap_hist_by_label.png")
    print(" - freeze_to_open_branch_stats.csv")
    print(" - freeze_to_open_events.csv")
    print(" - frozen_accounts_alert_reasons.csv")


if __name__ == "__main__":
    main()
