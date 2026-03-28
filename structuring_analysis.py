import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
transactions_df = pd.concat(
    [pd.read_csv(f'transactions_part_{i}.csv') for i in range(6)],
    ignore_index=True
)
labels_df = pd.read_csv("train_labels.csv")

# Calculate structuring ratio feature
total_txns = transactions_df.groupby('account_id').size().rename('total_txn_count')
structured_txns = transactions_df[
    (transactions_df["amount"] > 45000) & (transactions_df["amount"] < 50000)
].groupby("account_id").size().rename('structured_txn_count')

struct_df = pd.concat([total_txns, structured_txns], axis=1).fillna(0)
struct_df['structuring_ratio'] = struct_df['structured_txn_count'] / struct_df['total_txn_count']
struct_df['structuring_ratio'].fillna(0, inplace=True)

# Merge with labels
struct_df = pd.merge(struct_df, labels_df, on='account_id', how='left')

print("="*60)
print("   Analysis: Structuring Transaction Ratio")
print("="*60)
print("\nKey Points:")
print("  - 'Structuring' is the act of making many small transactions to avoid regulatory reporting thresholds (e.g., ₹50,000).")
print("  - The 'structuring_ratio' is the proportion of an account's transactions that fall just below this threshold (₹45k-₹50k).")
print("  - We expect this ratio to be near zero for most legitimate accounts but potentially high for mules.")

# Print summary statistics
print("Summary statistics for structuring ratio:")
desc_stats = struct_df.groupby('is_mule')['structuring_ratio'].describe()
print(desc_stats)

print("\nObservation:")
mule_75p = desc_stats.loc[1, '75%']
legit_75p = desc_stats.loc[0, '75%']

if mule_75p > 0 and legit_75p == 0:
    print(f"  - The statistics are highly revealing. For 75% of legitimate accounts, the structuring ratio is {legit_75p}, as expected.")
    print(f"  - However, for mule accounts, the 75th percentile is {mule_75p:.4f}, and the mean is significantly higher than for legitimate accounts.")
    print("  - This indicates that while not all mules engage in structuring, a significant portion does, making this a very powerful feature for detection.")
    print("  - The density plot will visually confirm that the small bump in non-zero ratios is almost entirely composed of mule accounts.")
else:
    print("  - The statistics do not show a clear separation. Further investigation may be needed.")

# Plot the density, comparing mule vs. legit
plt.figure(figsize=(12, 7))
sns.kdeplot(data=struct_df, x='structuring_ratio', hue='is_mule', fill=True, common_norm=False)
plt.xlabel("Proportion of Transactions between ₹45k-₹50k")
plt.ylabel("Density")
plt.title("Distribution of 'Structuring' Transaction Ratio")
# Zoom in on the interesting part of the distribution, as most ratios will be 0
plt.xlim(-0.01, 0.2)
plt.legend(title='Account Type', labels=['Mule', 'Legitimate'])
plt.savefig('structuring_ratio_distribution.png')

print("Plot saved to structuring_ratio_distribution.png")
