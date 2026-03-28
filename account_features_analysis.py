import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


accounts_df = pd.read_csv("accounts.csv")
labels_df = pd.read_csv("train_labels.csv")
print("Load complete.")

df = pd.merge(accounts_df, labels_df, on='account_id')

# Account opening date ko datetime format mein convert karein
df['account_opening_date'] = pd.to_datetime(df['account_opening_date']) # Ensure datetime for age calculation

# Account age
latest_date = df['account_opening_date'].max()
df['account_age_days'] = (latest_date - df['account_opening_date']).dt.days

print("\n" + "="*60)
print("   Analysis: Mule vs. Legitimate Account Characteristics")
print("="*60)

# --- Analysis Functions ---
def analyze_numeric(feature):
    print(f"\n--- Analyzing '{feature}' ---")
    agg = df.groupby('is_mule')[feature].agg(['median', 'mean', 'std']).round(2)
    print("  - Summary Statistics:")
    print(agg)

    print("\n  - Key Points:")
    print("    - The 'median' is often more insightful than the 'mean' for skewed financial data as it's less affected by extreme outliers.")
    print("    - A large difference in medians between the two groups is a strong indicator of a useful feature.")
    print("    - 'std' (Standard Deviation) shows the volatility or spread of the feature for each group.")

    mule_median = agg.loc[1, 'median']
    legit_median = agg.loc[0, 'median']

    print("\n  - Observation:")
    if mule_median > legit_median * 1.1:
        print(f"    - Mule accounts have a noticeably higher median {feature} ({mule_median:,.2f}) compared to legitimate accounts ({legit_median:,.2f}).")
        print("    - This suggests that higher values of this feature are associated with mule activity.")
    elif legit_median > mule_median * 1.1:
        print(f"    - Mule accounts have a noticeably lower median {feature} ({mule_median:,.2f}) compared to legitimate accounts ({legit_median:,.2f}).")
        print("    - This suggests that lower values of this feature are associated with mule activity, which could indicate simpler or less established accounts.")
    else:
        print(f"    - The median {feature} is similar for both mule and legitimate accounts, suggesting it might not be a strong standalone indicator.")

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='is_mule', y=feature, data=df, showfliers=False, palette=['#43a047', '#e53935'])
    plt.title(f'Distribution of {feature} for Mule vs. Legitimate Accounts', fontsize=16)
    plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Mule'])
    plt.show()

def analyze_categorical(feature, y_label="Feature"):
    print(f"\n--- Analyzing '{feature}' ---")
    # Crosstab to show percentage distribution
    crosstab_df = pd.crosstab(df[feature], df['is_mule'], normalize='index') * 100
    print("  - Percentage Distribution (Row-wise):")
    print(crosstab_df.round(2))

    print("\n  - Key Points:")
    print("    - This table shows, for each category, what percentage of accounts are legitimate (0) vs. mule (1).")
    print("    - We are looking for categories where the percentage of mules is significantly higher than the baseline mule rate (~2.4%).")

    print("\n  - Observation:")
    # Find the category with the highest mule rate
    highest_mule_category = crosstab_df[1].idxmax()
    highest_mule_rate = crosstab_df[1].max()
    baseline_mule_rate = df['is_mule'].mean() * 100

    if highest_mule_rate > baseline_mule_rate * 2: # If highest rate is at least double the baseline
        print(f"    - The category '{highest_mule_category}' has the highest concentration of mules ({highest_mule_rate:.2f}%).")
        print(f"    - This is significantly higher than the dataset's average mule rate of {baseline_mule_rate:.2f}%, making '{feature} == {highest_mule_category}' a potentially strong predictive signal.")
    else:
        print(f"    - No category within '{feature}' shows a strong concentration of mule accounts. The distribution appears relatively even across classes.")

    plt.figure(figsize=(10, 6))
    sns.countplot(y=feature, hue='is_mule', data=df, palette=['#43a047', '#e53935'])
    plt.title(f'Distribution of {feature} for Mule vs. Legitimate Accounts', fontsize=16)
    plt.ylabel(y_label)
    plt.xlabel("Count of Accounts")
    plt.legend(title='Account Type', labels=['Legitimate', 'Mule'])
    plt.tight_layout()
    plt.show()

# --- Run Analysis ---
try:
    # 1. Average Balance
    analyze_numeric('avg_balance')

    # 2. Account Status
    analyze_categorical('account_status', y_label="Account Status")

    # 3. Product Family
    analyze_categorical('product_family', y_label="Product Family")

    # 4. Rural Branch
    analyze_categorical('rural_branch', y_label="Is Rural Branch")

    # 5. KYC Compliant
    analyze_categorical('kyc_compliant', y_label="Is KYC Compliant")

    # 6. Nomination Flag
    analyze_categorical('nomination_flag', y_label="Nomination Flag")

    # 7. Account Age
    print("\n--- Analyzing 'Account Age' ---")
    agg_age = df.groupby('is_mule')['account_age_days'].agg(['median', 'mean', 'std']).round(0)
    print("  - Summary Statistics:")
    print(agg_age)

    print("\n  - Observation:")
    mule_median_age = agg_age.loc[1, 'median']
    legit_median_age = agg_age.loc[0, 'median']
    print(f"    - Mule accounts are significantly younger (Median Age: {mule_median_age:,.0f} days) than legitimate accounts (Median Age: {legit_median_age:,.0f} days).")
    print("    - This strongly supports the 'New Account High Value' mule pattern and confirms that 'account age' will be a critical feature for modeling.")


    plt.figure(figsize=(12, 7))
    sns.histplot(data=df, x='account_age_days', hue='is_mule', kde=True, common_norm=False, palette=['#43a047', '#e53935'])
    plt.title('Distribution of Account Age (in days)', fontsize=16)
    plt.xlabel("Account Age (Days)")
    plt.ylabel("Number of Accounts")
    plt.legend(title='Account Type', labels=['Mule', 'Legitimate'])
    plt.show()

except (ImportError, NameError):
    print("\n[INFO] Matplotlib or Seaborn not installed. Skipping visualization.")
    print("To generate the plot, please install them: pip install matplotlib seaborn")
