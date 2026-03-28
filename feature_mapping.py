import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

accounts_df = pd.read_csv("accounts.csv")

# Use train_labels.csv as the source for alerts and filter for mule accounts
alerts_df = pd.read_csv("train_labels.csv")
alerts_df = alerts_df[alerts_df['is_mule'] == 1].copy()
print("Load complete.")

# -----------------------------
# Alert -> Feature mapping
# -----------------------------
alert_feature_map = {
    "Income-Transaction Mismatch": [
        "avg_balance", "product_family", "kyc_compliant", "account_opening_date"
    ],
    "Layered Transaction Pattern": [
        "avg_balance", "product_family", "account_status", "kyc_compliant"
    ],
    "Post-Contact-Update Spike": [
        "account_opening_date", "kyc_compliant", "account_status"
    ],
    "Rapid Movement of Funds": [
        "avg_balance", "product_family", "account_opening_date"
    ],
    "Round Amount Pattern": [
        "avg_balance", "product_family", "account_status"
    ],
    "Routine Investigation": [
        "kyc_compliant", "nomination_flag", "account_status", "rural_branch"
    ],
    "Structuring Transactions Below Threshold": [
        "avg_balance", "product_family", "account_status"
    ],
    "Unusual Fund Flow Pattern": [
        "avg_balance", "account_status", "kyc_compliant"
    ],
    "Dormant Account Reactivation": [
        "avg_balance", "account_status", "kyc_compliant", "account_opening_date"
    ],
    "Geographic Anomaly Detected": [
        "rural_branch", "account_opening_date"
    ],
    "High-Value Activity on New Account": [
        "avg_balance", "kyc_compliant", "account_opening_date", "product_family"
    ]
}

# -----------------------------
# 3) Function to assign features
# -----------------------------
def get_features(reason):
   
    return ", ".join(alert_feature_map.get(reason, ["Unknown reason"]))

print("\nApplying feature mapping to alert reasons...")
alerts_df["important_features"] = alerts_df["alert_reason"].apply(get_features)
print("Mapping complete.")

# -----------------------------
# 4) Merge with accounts.csv to create a comprehensive dataset
# -----------------------------
print("Merging alerts with account details for analysis...")
merged_df = alerts_df.merge(accounts_df, how="left", on="account_id")
print("Merge complete.")

# -----------------------------
# 5) Output and Analysis (No file saving)
# -----------------------------
print("\n===== Sample of Alerts with Mapped Important Features =====\n")
print(merged_df[["account_id", "alert_reason", "important_features"]].head(10).to_string())

print("\n" + "="*60)
print("   Visual Analysis for Top Alert Reason (as requested)")
print("="*60)

# --- Analysis & Visualization ---
try:
    # Find the most common alert reason
    top_reason = merged_df['alert_reason'].mode()[0]
    print(f"\nAnalyzing the most common alert reason: '{top_reason}'")

    # Filter the dataframe for this reason
    top_reason_df = merged_df[merged_df['alert_reason'] == top_reason]

    # The mapping suggests looking at 'avg_balance' for this reason. Let's analyze it.
    feature_to_analyze = 'avg_balance'
    print(f"\nDescriptive statistics for '{feature_to_analyze}' for this group:")
    print(top_reason_df[feature_to_analyze].describe().round(2).to_string())

    plt.figure(figsize=(12, 7))
    sns.histplot(top_reason_df[feature_to_analyze], kde=True, bins=15, color='#4e79a7')
    median_val = top_reason_df[feature_to_analyze].median()
    plt.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median Balance: {median_val:,.0f}')
    plt.title(f"Distribution of '{feature_to_analyze}' for Mule Accounts flagged with\n'{top_reason}'", fontsize=16)
    plt.xlabel("Average Balance", fontsize=12)
    plt.ylabel("Number of Accounts", fontsize=12)
    plt.legend()
    plt.ticklabel_format(style='plain', axis='x')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    print(f"\n[SUCCESS] A graph has been generated for the '{feature_to_analyze}' feature.")
    print("\nInsight: This plot shows the specific balance distribution for mule accounts flagged due to an income/transaction mismatch. This kind of targeted analysis can reveal patterns specific to certain fraud types.")

except (ImportError, NameError):
    print("\n[INFO] Matplotlib or Seaborn not installed. Skipping visualization.")
    print("To generate the plot, please install them: pip install matplotlib seaborn")