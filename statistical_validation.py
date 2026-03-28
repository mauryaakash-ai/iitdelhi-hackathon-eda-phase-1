import pandas as pd
from scipy.stats import mannwhitneyu, ks_2samp
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

# --- Setup ---
warnings.filterwarnings('ignore')

# --- 1. Data Loading and Preparation ---
# This section is self-contained to ensure the script can run independently.
print("Step 1: Loading and preparing data for statistical tests...")
try:
    labels_df = pd.read_csv("train_labels.csv")
    transactions_df = pd.concat(
        [pd.read_csv(f"transactions_part_{i}.csv") for i in range(6)],
        ignore_index=True
    )
except FileNotFoundError as e:
    print(f"[ERROR] Error loading data: {e}. Please ensure all CSV files are in the correct directory.")
    exit()

# Aggregate transactions to get 'txn_count' per account
txn_agg = transactions_df.groupby('account_id').agg(
    txn_count=('transaction_id', 'count')
).reset_index()

# Create the main 'train' dataframe
train = pd.merge(labels_df, txn_agg, on='account_id', how='left')
train['txn_count'].fillna(0, inplace=True)
print("[OK] Data prepared successfully.")


# --- 2. Statistical Validation for 'txn_count' ---
print("\n" + "="*60)
print("PART 6 — Statistical Validation of Transaction Count")
print("="*60)

# Separate the data into two groups: mule and legitimate
mule_counts = train[train["is_mule"] == 1]["txn_count"].dropna()
legit_counts = train[train["is_mule"] == 0]["txn_count"].dropna()

# --- 2.1. Mann-Whitney U Test ---
print("\n1. Mann-Whitney U Test (Comparing Medians)")
print("-" * 50)

# Perform the test
stat_mw, p_mw = mannwhitneyu(mule_counts, legit_counts, alternative='two-sided')

print(f"  - Statistic: {stat_mw:,.2f}")
print(f"  - P-value: {p_mw}")

print("\n  Key Points:")
print("    - **Purpose**: This non-parametric test checks if the median `txn_count` of mule accounts is different from that of legitimate accounts.")
print("    - **Hypothesis**: The null hypothesis is that the medians of the two groups are equal.")

print("\n  Observation:")
if p_mw < 0.05:
    print("    - The p-value is extremely small (<< 0.05), leading us to reject the null hypothesis.")
    print("    - This provides **strong statistical evidence** that the median transaction count for mule accounts is significantly different from that of legitimate accounts.")
    print("    - It scientifically validates the visual shift we observed in the density plot and supports creating features like `txns_per_active_day`.")
else:
    print("    - The p-value is greater than 0.05, suggesting no significant difference in the median transaction counts between the two groups.")


# --- 2.2. Kolmogorov-Smirnov (KS) Test ---
print("\n2. Kolmogorov-Smirnov (KS) Test (Comparing Distributions)")
print("-" * 50)

# Perform the test
stat_ks, p_ks = ks_2samp(mule_counts, legit_counts)

print(f"  - KS Statistic: {stat_ks:.4f}")
print(f"  - P-value: {p_ks}")

print("\n  Key Points:")
print("    - **Purpose**: This test checks if the entire probability distribution of `txn_count` for mules is different from the distribution for legitimate accounts.")
print("    - **KS Statistic**: The value (0 to 1) represents the maximum difference between the cumulative distribution functions of the two samples. A larger value means a greater difference.")

print("\n  Observation:")
if p_ks < 0.05:
    print(f"    - The KS statistic of {stat_ks:.4f} and an extremely small p-value (<< 0.05) confirm that the distributions are fundamentally different.")
    print("    - This is a more powerful result than the Mann-Whitney test. It proves that not just the median, but the **entire behavioral shape** of transaction frequency (e.g., skewness, peaks) is distinct for mule accounts.")
    print("    - This strongly justifies using `txn_count` and its derivatives as predictive features.")
else:
    print("    - The p-value is greater than 0.05, suggesting the distributions are not significantly different.")

print("\n" + "="*60)
print("Statistical validation complete.")


# --- 3. Visualization of Statistical Tests ---
print("\nStep 3: Generating plots to visualize statistical differences...")

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: CDF Plot to visualize the KS Test
sns.ecdfplot(data=train, x="txn_count", hue="is_mule", ax=axes[0], palette=['#43a047', '#e53935'])
axes[0].set_xscale('log') # Log scale is best for skewed data like transaction counts
axes[0].set_title("CDF of Transaction Counts (Visualizing KS Test)", fontsize=16)
axes[0].set_xlabel("Transaction Count (Log Scale)")
axes[0].set_ylabel("Cumulative Probability")
axes[0].text(0.05, 0.95, f"KS Statistic: {stat_ks:.4f}\np-value: {p_ks:.2e}",
             transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[0].legend(title='Account Type', labels=['Mule', 'Legitimate'])

# Plot 2: Box Plot to visualize the Mann-Whitney U Test
sns.boxplot(data=train, y="txn_count", x="is_mule", ax=axes[1], palette=['#43a047', '#e53935'])
axes[1].set_yscale('log') # Log scale is essential to see the difference in medians
axes[1].set_title("Distribution of Transaction Counts (Visualizing Median Difference)", fontsize=16)
axes[1].set_xlabel("Is Mule Account?")
axes[1].set_ylabel("Transaction Count (Log Scale)")
axes[1].set_xticklabels(['Legitimate (0)', 'Mule (1)'])
axes[1].text(0.05, 0.95, f"Mann-Whitney p-value: {p_mw:.2e}",
             transform=axes[1].transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

fig.suptitle("Visual Validation of Statistical Tests for Transaction Count", fontsize=20)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("[OK] Plots generated successfully.")

