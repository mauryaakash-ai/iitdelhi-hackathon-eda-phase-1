import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
accounts = pd.read_csv('accounts.csv')
transactions = pd.concat([pd.read_csv(f'transactions_part_{i}.csv') for i in range(6)], ignore_index=True)
labels = pd.read_csv('train_labels.csv')

# Merge dataframes
df = pd.merge(transactions, accounts, on='account_id')
df = pd.merge(df, labels, on='account_id')

# Convert dates to datetime objects
df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
df['account_opening_date'] = pd.to_datetime(df['account_opening_date'])

# Calculate account age at the time of transaction
df['account_age_at_txn'] = (df['transaction_timestamp'] - df['account_opening_date']).dt.days

# Define "new account" period
new_account_period = 30

# Filter for transactions within the new account period
new_account_txns = df[df['account_age_at_txn'] <= new_account_period]

# Aggregate data for new accounts
new_account_agg = new_account_txns.groupby('account_id').agg(
    total_txn_value=('amount', 'sum'),
    txn_count=('transaction_id', 'count'),
    is_mule=('is_mule', 'first')
).reset_index()

# Print summary statistics
print("Summary statistics for total transaction value within 30 days:")
print(new_account_agg.groupby('is_mule')['total_txn_value'].describe())
print("Summary statistics for transaction count within 30 days:")
print(new_account_agg.groupby('is_mule')['txn_count'].describe())

# Create and save plots
plt.figure(figsize=(12, 6))
sns.boxplot(x='is_mule', y='total_txn_value', data=new_account_agg, showfliers=False, hue='is_mule')
plt.title(f'Total Transaction Value within {new_account_period} days of Account Opening')
plt.xlabel('Is Mule')
plt.ylabel('Total Transaction Value')
plt.savefig('new_account_txn_value.png')

plt.figure(figsize=(12, 6))
sns.boxplot(x='is_mule', y='txn_count', data=new_account_agg, showfliers=False, hue='is_mule')
plt.title(f'Transaction Count within {new_account_period} days of Account Opening')
plt.xlabel('Is Mule')
plt.ylabel('Transaction Count')
plt.savefig('new_account_txn_count.png')

print("Plots saved to new_account_txn_value.png and new_account_txn_count.png")
