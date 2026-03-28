import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

print("Starting comprehensive feature visualization pipeline...")

# --- 1. Data Loading ---
print("1. Loading raw data...")
try:
    labels_df = pd.read_csv("train_labels.csv")
    accounts_df = pd.read_csv("accounts.csv")
    customers_df = pd.read_csv("customers.csv")
    customer_account_linkage_df = pd.read_csv("customer_account_linkage.csv")
    transactions_df = pd.concat(
        [pd.read_csv(f"transactions_part_{i}.csv") for i in range(6)],
        ignore_index=True
    )
    print("[OK] All data loaded successfully.")
except FileNotFoundError as e:
    print(f"[ERROR] Error loading data: {e}. Please ensure all CSV files are in the correct directory.")
    exit()

# --- 2. Initial Data Preprocessing ---
print("2. Initial data preprocessing (datetime conversions, merging labels)...")
transactions_df['transaction_timestamp'] = pd.to_datetime(transactions_df['transaction_timestamp'])
accounts_df['account_opening_date'] = pd.to_datetime(accounts_df['account_opening_date'])
customers_df['date_of_birth'] = pd.to_datetime(customers_df['date_of_birth'])
customers_df['relationship_start_date'] = pd.to_datetime(customers_df['relationship_start_date'])
if 'last_mobile_update_date' in customers_df.columns:
    customers_df['last_mobile_update_date'] = pd.to_datetime(customers_df['last_mobile_update_date'], errors='coerce')
labels_df['mule_flag_date'] = pd.to_datetime(labels_df['mule_flag_date'])

# Merge labels early to filter transactions for leakage prevention
accounts_with_labels = pd.merge(accounts_df, labels_df, on='account_id', how='left')

# Apply data leakage prevention: Filter transactions to only include those BEFORE the mule_flag_date
transactions_filtered = pd.merge(transactions_df, labels_df[['account_id', 'mule_flag_date']], on='account_id', how='left')
transactions_filtered = transactions_filtered[
    (transactions_filtered['mule_flag_date'].isna()) |
    (transactions_filtered['transaction_timestamp'] < transactions_filtered['mule_flag_date'])
]
transactions_filtered.drop(columns=['mule_flag_date'], inplace=True)

print("[OK] Preprocessing complete and data leakage prevented.")

# --- 3. Feature Calculation (Identical to feature_engineering_pipeline.py) ---
print("3. Calculating features for visualization...")

features_df = accounts_with_labels[['account_id', 'is_mule']].copy()

# Centralized Transaction Aggregations
account_summary = transactions_filtered.groupby('account_id').agg(
    total_txn_count=('transaction_id', 'count'),
    credit_txn_count=('amount', lambda x: (transactions_filtered.loc[x.index, 'txn_type'] == 'C').sum()),
    debit_txn_count=('amount', lambda x: (transactions_filtered.loc[x.index, 'txn_type'] == 'D').sum()),
    sum_credits=('amount', lambda x: x[transactions_filtered.loc[x.index, 'txn_type'] == 'C'].sum()),
    sum_debits=('amount', lambda x: x[transactions_filtered.loc[x.index, 'txn_type'] == 'D'].sum()),
    avg_transaction_amount=('amount', 'mean'),
    max_transaction_amount=('amount', 'max')
).reset_index()
features_df = pd.merge(features_df, account_summary, on='account_id', how='left')

# 3.2. Structuring
structured_txns = transactions_filtered[
    (transactions_filtered["amount"] > 45000) & (transactions_filtered["amount"] < 50000)
].groupby("account_id").size().rename('structured_txn_count')
struct_features = structured_txns.reset_index()
features_df = pd.merge(features_df, struct_features, on='account_id', how='left')
features_df['structured_txn_count'].fillna(0, inplace=True)
features_df['structuring_ratio'] = features_df['structured_txn_count'] / features_df['total_txn_count']

# 3.3. Rapid Pass-Through
transactions_filtered.sort_values(['account_id', 'transaction_timestamp'], inplace=True)
def calculate_pass_through(group):
    score = 0
    time_threshold = pd.Timedelta(days=1)
    amount_threshold = 0.95
    for i in range(len(group) - 1):
        current_txn = group.iloc[i]
        next_txn = group.iloc[i+1]
        if current_txn['txn_type'] == 'C' and next_txn['txn_type'] == 'D':
            time_diff = next_txn['transaction_timestamp'] - current_txn['transaction_timestamp']
            if time_diff <= time_threshold and next_txn['amount'] >= amount_threshold * current_txn['amount']:
                score += 1
    return score
pass_through_scores = transactions_filtered.groupby('account_id').apply(calculate_pass_through)
pass_through_features = pass_through_scores.rename('pass_through_score').reset_index()
features_df = pd.merge(features_df, pass_through_features, on='account_id', how='left')

# 3.4. Account Age & Basic Account Info
latest_date_in_data = transactions_df['transaction_timestamp'].max()
accounts_df['account_age_days'] = (latest_date_in_data - accounts_df['account_opening_date']).dt.days
features_df = pd.merge(features_df, accounts_df[['account_id', 'account_age_days', 'avg_balance', 'account_status', 'product_family', 'branch_code', 'rural_branch', 'kyc_compliant', 'nomination_flag']], on='account_id', how='left')

# 3.5. Dormant Activation
transactions_filtered['days_since_last_txn'] = transactions_filtered.groupby('account_id')['transaction_timestamp'].diff().dt.days
dormancy_features = transactions_filtered.groupby('account_id')['days_since_last_txn'].max().rename('max_dormancy_days').reset_index()
features_df = pd.merge(features_df, dormancy_features, on='account_id', how='left')
def calculate_dormant_activation(group):
    if len(group) < 2 or group['days_since_last_txn'].isnull().all():
        return pd.Series({'post_dormancy_burst_volume': 0, 'post_dormancy_burst_count': 0})
    max_gap_end_idx = group['days_since_last_txn'].idxmax()
    dormancy_end_date = group.loc[max_gap_end_idx, 'transaction_timestamp']
    burst_window_end = dormancy_end_date + pd.Timedelta(days=30)
    burst_txns = group[
        (group['transaction_timestamp'] >= dormancy_end_date) &
        (group['transaction_timestamp'] <= burst_window_end)
    ]
    return pd.Series({
        'post_dormancy_burst_volume': burst_txns['amount'].sum(),
        'post_dormancy_burst_count': len(burst_txns)
    })
dormant_activation_features = transactions_filtered.groupby('account_id').apply(calculate_dormant_activation).reset_index()
features_df = pd.merge(features_df, dormant_activation_features, on='account_id', how='left')

# 3.6. Fan-In / Fan-Out
counterparty_features = transactions_filtered.groupby('account_id')['counterparty_id'].nunique().rename('unique_counterparties').reset_index()
credit_counterparties = transactions_filtered[transactions_filtered['txn_type'] == 'C'].groupby('account_id')['counterparty_id'].nunique().rename('unique_credit_counterparties').reset_index()
debit_counterparties = transactions_filtered[transactions_filtered['txn_type'] == 'D'].groupby('account_id')['counterparty_id'].nunique().rename('unique_debit_counterparties').reset_index()
features_df = pd.merge(features_df, counterparty_features, on='account_id', how='left')
features_df = pd.merge(features_df, credit_counterparties, on='account_id', how='left')
features_df = pd.merge(features_df, debit_counterparties, on='account_id', how='left')
features_df['fan_in_ratio'] = features_df['unique_credit_counterparties'] / features_df['credit_txn_count']
features_df['fan_out_ratio'] = features_df['unique_debit_counterparties'] / features_df['debit_txn_count']

# 3.7. Geographic Anomaly
customer_info = pd.merge(customer_account_linkage_df, customers_df, on='customer_id', how='left')
account_geo_info = pd.merge(accounts_df[['account_id', 'branch_code', 'branch_pin']], customer_info[['account_id', 'customer_pin']], on='account_id', how='left')
account_geo_info['branch_customer_pin_mismatch'] = (account_geo_info['branch_pin'] != account_geo_info['customer_pin']).astype(int)
features_df = pd.merge(features_df, account_geo_info[['account_id', 'branch_customer_pin_mismatch']], on='account_id', how='left')

# 3.8. New Account High Value
txns_with_open_date = pd.merge(transactions_filtered, accounts_df[['account_id', 'account_opening_date']], on='account_id', how='left')
first_30d_txns = txns_with_open_date[
    txns_with_open_date['transaction_timestamp'] <= (txns_with_open_date['account_opening_date'] + pd.Timedelta(days=30))
]
new_acct_features = first_30d_txns.groupby('account_id').agg(
    total_txn_value_first_30d=('amount', 'sum'),
    txn_count_first_30d=('transaction_id', 'count')
).reset_index()
features_df = pd.merge(features_df, new_acct_features, on='account_id', how='left')

# 3.9. Income Mismatch
features_df['credit_turnover_to_balance_ratio'] = features_df['sum_credits'] / (features_df['avg_balance'] + 1e-6)
features_df['max_txn_to_balance_ratio'] = features_df['max_transaction_amount'] / (features_df['avg_balance'] + 1e-6)

# 3.10. Post-Mobile-Change Spike
if 'last_mobile_update_date' in customers_df.columns:
    customer_mobile_updates = customers_df[['customer_id', 'last_mobile_update_date']].dropna()
    transactions_with_customer = pd.merge(transactions_filtered, customer_account_linkage_df, on='account_id', how='left')
    transactions_with_customer = pd.merge(transactions_with_customer, customer_mobile_updates, on='customer_id', how='left')
    def calculate_post_mobile_spike(group):
        if group['last_mobile_update_date'].isnull().all():
            return pd.Series({'post_mobile_change_txn_count': 0, 'post_mobile_change_txn_value': 0})
        update_date = group['last_mobile_update_date'].dropna().iloc[0]
        post_update_txns = group[group['transaction_timestamp'] > update_date]
        window_end = update_date + pd.Timedelta(days=30)
        post_update_txns_window = post_update_txns[post_update_txns['transaction_timestamp'] <= window_end]
        return pd.Series({
            'post_mobile_change_txn_count': len(post_update_txns_window),
            'post_mobile_change_txn_value': post_update_txns_window['amount'].sum()
        })
    post_mobile_spike_features = transactions_with_customer.groupby('account_id').apply(calculate_post_mobile_spike).reset_index()
    features_df = pd.merge(features_df, post_mobile_spike_features, on='account_id', how='left')
else:
    print("  - Skipping 'Post-Mobile-Change Spike' features: 'last_mobile_update_date' column not found.")
    features_df['post_mobile_change_txn_count'] = 0
    features_df['post_mobile_change_txn_value'] = 0

# 3.11. Round Amount Patterns
round_amount_thresholds = [1000, 5000, 10000, 50000]
total_txns_series = features_df.set_index('account_id')['total_txn_count']
for threshold in round_amount_thresholds:
    col_name = f'round_amount_ratio_{threshold}'
    round_counts = transactions_filtered[
        (transactions_filtered['amount'] > 0) & (transactions_filtered['amount'] % threshold == 0)
    ].groupby('account_id').size()
    features_df[col_name] = features_df['account_id'].map(round_counts) / features_df['account_id'].map(total_txns_series)

# 3.12. Salary Cycle Exploitation
transactions_filtered['day_of_month'] = transactions_filtered['transaction_timestamp'].dt.day
def calculate_salary_cycle_features(group):
    salary_window_txns = group[
        (group['day_of_month'] >= 25) | (group['day_of_month'] <= 5)
    ]
    total_txns_count = len(group)
    salary_window_count = len(salary_window_txns)
    if total_txns_count == 0:
        return pd.Series({'salary_cycle_activity_ratio': 0})
    return pd.Series({
        'salary_cycle_activity_ratio': salary_window_count / total_txns_count
    })
salary_cycle_features = transactions_filtered.groupby('account_id').apply(calculate_salary_cycle_features).reset_index()
features_df = pd.merge(features_df, salary_cycle_features, on='account_id', how='left')

# 3.13. Branch-Level Collusion
branch_mule_rates = accounts_with_labels.groupby('branch_code')['is_mule'].mean().rename('branch_mule_rate').reset_index()
features_df = pd.merge(features_df, branch_mule_rates, on='branch_code', how='left', suffixes=('', '_y'))

# Final cleaning for plotting
features_df.fillna(0, inplace=True)

print("[OK] Features calculated.")

# --- 4. Plotting Section ---
print("4. Generating plots for each feature...")

plot_features = [
    ('max_dormancy_days', 'Dormant Activation: Max Dormancy Days', 'Boxplot', False),
    ('post_dormancy_burst_volume', 'Dormant Activation: Post-Dormancy Burst Volume', 'Boxplot', True),
    ('post_dormancy_burst_count', 'Dormant Activation: Post-Dormancy Burst Count', 'Boxplot', True),
    ('structuring_ratio', 'Structuring: Ratio of Transactions near Threshold', 'KDEplot', False),
    ('pass_through_score', 'Rapid Pass-Through Score', 'Boxplot', True),
    ('account_age_days', 'Account Age (Days)', 'Boxplot', False),
    ('branch_customer_pin_mismatch', 'Geographic Anomaly: Branch vs Customer PIN Mismatch', 'Countplot', False),
    ('total_txn_value_first_30d', 'New Account High Value: Total Txn Value (First 30 Days)', 'Boxplot', True),
    ('txn_count_first_30d', 'New Account High Value: Txn Count (First 30 Days)', 'Boxplot', True),
    ('credit_turnover_to_balance_ratio', 'Income Mismatch: Credit Turnover to Avg Balance Ratio', 'Boxplot', True),
    ('max_txn_to_balance_ratio', 'Income Mismatch: Max Txn to Avg Balance Ratio', 'Boxplot', True),
    ('post_mobile_change_txn_count', 'Post-Mobile-Change Spike: Txn Count After Mobile Update', 'Boxplot', True),
    ('post_mobile_change_txn_value', 'Post-Mobile-Change Spike: Txn Value After Mobile Update', 'Boxplot', True),
    ('round_amount_ratio_1000', 'Round Amount Pattern: Ratio of 1K Round Amounts', 'KDEplot', False),
    ('round_amount_ratio_50000', 'Round Amount Pattern: Ratio of 50K Round Amounts', 'KDEplot', False),
    ('salary_cycle_activity_ratio', 'Salary Cycle Exploitation: Activity Ratio in Salary Window', 'KDEplot', False),
    ('branch_mule_rate', 'Branch-Level Collusion: Mule Rate per Branch', 'Boxplot', False)
]

categorical_features = ['account_status', 'product_family', 'rural_branch', 'kyc_compliant', 'nomination_flag']

# Plotting categorical features
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(y=feature, hue='is_mule', data=features_df, palette=['#43a047', '#e53935'])
    plt.title(f'Distribution of {feature} for Mule vs. Legitimate Accounts')
    plt.xlabel("Count of Accounts")
    plt.ylabel(feature.replace('_', ' ').title())
    plt.legend(title='Account Type', labels=['Legitimate', 'Mule'])
    plt.tight_layout()
    plt.savefig(f'plots/{feature}_distribution.png')
    plt.close()
    print(f"  - Plot saved for {feature}")


for feature, title, plot_type, log_scale in plot_features:
    plt.figure(figsize=(10, 6))
    plot_data = features_df.copy()

    if log_scale:
        # Add 1 before log to handle zero values
        plot_data[feature] = np.log1p(plot_data[feature])
        title = f"Log(1 + {title})"

    if plot_type == 'Boxplot':
        sns.boxplot(x='is_mule', y=feature, data=plot_data, showfliers=False, palette=['#43a047', '#e53935'])
        plt.ylabel(title)
    elif plot_type == 'KDEplot':
        sns.kdeplot(data=plot_data, x=feature, hue='is_mule', fill=True, common_norm=False, palette=['#43a047', '#e53935'])
        plt.xlabel(title)
        plt.ylabel("Density")
    elif plot_type == 'Countplot':
        sns.countplot(x=feature, hue='is_mule', data=plot_data, palette=['#43a047', '#e53935'])
        plt.xlabel(title)
        plt.ylabel("Count")

    plt.title(f'Distribution of {title} (Mule vs. Legitimate)')
    plt.xticks(ticks=[0, 1], labels=['Legitimate', 'Mule'])
    plt.legend(title='Account Type', labels=['Legitimate', 'Mule'])
    plt.tight_layout()
    plt.savefig(f'plots/{feature}_distribution.png')
    plt.close()
    print(f"  - Plot saved for {feature}")

print("\n[OK] All plots generated and saved to the 'plots/' directory.")