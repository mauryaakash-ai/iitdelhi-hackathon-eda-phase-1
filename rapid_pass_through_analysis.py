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

# Separate credits and debits
credits = transactions_df[transactions_df['txn_type'] == 'C'].copy()
debits = transactions_df[transactions_df['txn_type'] == 'D'].copy()

# Rename columns for merging
credits.rename(columns={'amount': 'credit_amount', 'transaction_timestamp': 'credit_timestamp'}, inplace=True)
debits.rename(columns={'amount': 'debit_amount', 'transaction_timestamp': 'debit_timestamp'}, inplace=True)

# Merge credits and debits on account_id
merged = pd.merge(credits, debits, on='account_id')

# Filter for rapid pass-through patterns
time_threshold = pd.Timedelta(days=1)
amount_threshold = 0.95  # Debits that are at least 95% of the credit

rapid_pass_through = merged[
    (merged['debit_timestamp'] > merged['credit_timestamp']) &
    (merged['debit_timestamp'] - merged['credit_timestamp'] <= time_threshold) &
    (merged['debit_amount'] >= amount_threshold * merged['credit_amount'])
]

# Calculate rapid pass-through score
pass_through_score = rapid_pass_through.groupby('account_id').size().rename('pass_through_score').reset_index()

# Merge with labels
pass_through_score = pd.merge(pass_through_score, labels_df, on='account_id', how='right')
pass_through_score['pass_through_score'].fillna(0, inplace=True)

# Print summary statistics
print("Summary statistics for rapid pass-through score:")
print(pass_through_score.groupby('is_mule')['pass_through_score'].describe())

# Plot the distribution of the score
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.boxplot(x='is_mule', y='pass_through_score', data=pass_through_score, showfliers=False, hue='is_mule')
plt.title('Distribution of Rapid Pass-Through Score')
plt.xlabel('Is Mule')
plt.ylabel('Rapid Pass-Through Score')
plt.savefig('rapid_pass_through_score.png')

print("Plot saved to rapid_pass_through_score.png")
