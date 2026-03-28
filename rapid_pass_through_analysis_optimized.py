import pandas as pd
import numpy as np

# Load data
transactions_df = pd.concat(
    [pd.read_csv(f'transactions_part_{i}.csv') for i in range(6)],
    ignore_index=True
)
labels_df = pd.read_csv("train_labels.csv")

# Convert timestamp to datetime
transactions_df['transaction_timestamp'] = pd.to_datetime(transactions_df['transaction_timestamp'])

# --- OPTIMIZED LOGIC TO AVOID MEMORY ERROR ---
# The previous pd.merge() approach creates a cartesian product for each account,
# leading to a MemoryError on large datasets.
# The correct, scalable approach is to sort and iterate.

# Sort transactions to process them chronologically for each account
transactions_df.sort_values(['account_id', 'transaction_timestamp'], inplace=True)

def calculate_pass_through(group):
    """
    Efficiently calculates the pass-through score for a single account's transactions.
    This avoids a computationally expensive cross-join.
    """
    score = 0
    time_threshold = pd.Timedelta(days=1)
    amount_threshold = 0.95

    # Iterate through transactions, stopping before the last one
    for i in range(len(group) - 1):
        current_txn = group.iloc[i]
        next_txn = group.iloc[i+1]

        # Check for a Credit followed by a Debit
        if current_txn['txn_type'] == 'C' and next_txn['txn_type'] == 'D':
            time_diff = next_txn['transaction_timestamp'] - current_txn['transaction_timestamp']

            # Check if the pattern matches the thresholds
            if time_diff <= time_threshold and next_txn['amount'] >= amount_threshold * current_txn['amount']:
                score += 1
    return score

print("Calculating rapid pass-through score efficiently (replaces inefficient cross-merge)...")
pass_through_scores = transactions_df.groupby('account_id').apply(calculate_pass_through)
pass_through_score = pass_through_scores.rename('pass_through_score').reset_index()
print("Calculation complete.")

# Merge with labels
pass_through_score = pd.merge(pass_through_score, labels_df, on='account_id', how='right')
pass_through_score['pass_through_score'].fillna(0, inplace=True)

# --- ANALYSIS AND OBSERVATION ---
print("\n" + "="*60)
print("   Analysis: Rapid Pass-Through Score")
print("="*60)

print("\nCode Implementation Note:")
print("  - The original cross-merge logic was computationally expensive (O(N*M) per account) and caused a MemoryError.")
print("  - Re-implemented by sorting transactions and iterating (O(T log T)), which is vastly more efficient and scalable.")

print("\nKey Points:")
print("  - 'Rapid Pass-Through' is a classic muling pattern where funds are deposited (credit) and then quickly withdrawn (debit).")
print("  - The 'pass_through_score' counts how many times a credit is followed by a >=95% matching debit within 24 hours.")
print("  - This behavior is highly anomalous for normal account usage, where funds typically rest for some time.")

# Print summary statistics
print("\nSummary statistics for rapid pass-through score:")
desc_stats = pass_through_score.groupby('is_mule')['pass_through_score'].describe()
print(desc_stats)

print("\nObservation:")
# Use .loc to safely access rows, even if one class is missing
if 1 in desc_stats.index and 0 in desc_stats.index:
    mule_mean = desc_stats.loc[1, 'mean']
    legit_mean = desc_stats.loc[0, 'mean']

    if mule_mean > legit_mean * 10: # A high multiplier because this is a rare event for legit accounts
        print(f"  - The difference is stark. The mean score for mule accounts ({mule_mean:.2f}) is orders of magnitude higher than for legitimate accounts ({legit_mean:.4f}).")
        print(f"  - The 75th percentile for legitimate accounts is {desc_stats.loc[0, '75%']:.0f}, while for mules it is {desc_stats.loc[1, '75%']:.0f}.")
        print("  - This confirms that rapid pass-through is a hallmark behavior of the mule accounts in this dataset and a critical feature for any detection model.")
    else:
        print("  - The statistics do not show a clear separation. The thresholds for time or amount may need tuning.")
else:
    print("  - Could not compare mule and legit stats; one or both groups may be missing from the data.")

# Plot the distribution of the score
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(x='is_mule', y='pass_through_score', data=pass_through_score, showfliers=False, palette=['#43a047', '#e53935'])
plt.title('Distribution of Rapid Pass-Through Score (Mule vs. Legitimate)')
plt.xlabel('Is Mule')
plt.ylabel('Rapid Pass-Through Score')
plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Mule'])
plt.savefig('rapid_pass_through_score_optimized.png')

print("\nPlot saved to rapid_pass_through_score_optimized.png")
