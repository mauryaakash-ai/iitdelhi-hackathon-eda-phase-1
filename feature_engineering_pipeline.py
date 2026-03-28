import pandas as pd
import numpy as np

print("Starting comprehensive feature engineering pipeline...")

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
    customers_df['last_mobile_update_date'] = pd.to_datetime(customers_df['last_mobile_update_date'], errors='coerce') # Use errors='coerce' to handle invalid date formats gracefully
labels_df['mule_flag_date'] = pd.to_datetime(labels_df['mule_flag_date'])

# Merge labels early to filter transactions for leakage prevention
accounts_with_labels = pd.merge(accounts_df, labels_df, on='account_id', how='left')

# Apply data leakage prevention: Filter transactions to only include those BEFORE the mule_flag_date
# For non-mule accounts (is_mule=0 or NaN), we keep all transactions.
# For mule accounts (is_mule=1), we keep transactions only up to the mule_flag_date.
transactions_filtered = pd.merge(transactions_df, labels_df[['account_id', 'mule_flag_date']], on='account_id', how='left')
transactions_filtered = transactions_filtered[
    (transactions_filtered['mule_flag_date'].isna()) |
    (transactions_filtered['transaction_timestamp'] < transactions_filtered['mule_flag_date'])
]

# Drop the temporary mule_flag_date column from transactions_filtered
transactions_filtered.drop(columns=['mule_flag_date'], inplace=True)

print("[OK] Preprocessing complete and data leakage prevented.")

# --- 3. Feature Engineering ---
print("3. Generating features from various patterns...")

# Initialize a DataFrame to hold all features, starting with account_id and is_mule
features_df = accounts_with_labels[['account_id', 'is_mule']].copy()

# --- 3.1 Centralized Transaction Aggregations (for efficiency) ---
print("  - Generating centralized transaction aggregations...")

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

# --- 3.2. Structuring (from structuring_analysis.py) ---
print("  - Generating feature: Structuring...")
structured_txns = transactions_filtered[
    (transactions_filtered["amount"] > 45000) & (transactions_filtered["amount"] < 50000)
].groupby("account_id").size().rename('structured_txn_count')

struct_features = structured_txns.reset_index()
features_df = pd.merge(features_df, struct_features, on='account_id', how='left')
features_df['structured_txn_count'].fillna(0, inplace=True)

# Use the pre-calculated total_txn_count for the ratio
features_df['structuring_ratio'] = features_df['structured_txn_count'] / features_df['total_txn_count']

# --- 3.3. Rapid Pass-Through (from rapid_pass_through_analysis_optimized.py) ---
print("  - Generating feature: Rapid Pass-Through...")

# Sort values for efficient iteration
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

# --- 3.4. Account Age & Basic Features (from account_features_analysis.py) ---
print("  - Generating features: Account Age & Basic Account Info...")
latest_date_in_data = transactions_df['transaction_timestamp'].max() # Use max transaction date as "today"
accounts_df['account_age_days'] = (latest_date_in_data - accounts_df['account_opening_date']).dt.days
features_df = pd.merge(features_df, accounts_df[['account_id', 'account_age_days', 'avg_balance', 'account_status', 'product_family', 'branch_code', 'rural_branch', 'kyc_compliant', 'nomination_flag']], on='account_id', how='left')

# --- 3.5. Dormant Activation ---
print("  - Generating features: Dormant Activation...")
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

# --- 3.6. Fan-In / Fan-Out ---
print("  - Generating features: Fan-In / Fan-Out...")
counterparty_features = transactions_filtered.groupby('account_id')['counterparty_id'].nunique().rename('unique_counterparties').reset_index()
credit_counterparties = transactions_filtered[transactions_filtered['txn_type'] == 'C'].groupby('account_id')['counterparty_id'].nunique().rename('unique_credit_counterparties').reset_index()
debit_counterparties = transactions_filtered[transactions_filtered['txn_type'] == 'D'].groupby('account_id')['counterparty_id'].nunique().rename('unique_debit_counterparties').reset_index()

features_df = pd.merge(features_df, counterparty_features, on='account_id', how='left')
features_df = pd.merge(features_df, credit_counterparties, on='account_id', how='left')
features_df = pd.merge(features_df, debit_counterparties, on='account_id', how='left')

# Correctly calculate ratios using credit/debit specific counts
features_df['fan_in_ratio'] = features_df['unique_credit_counterparties'] / features_df['credit_txn_count']
features_df['fan_out_ratio'] = features_df['unique_debit_counterparties'] / features_df['debit_txn_count']

# --- 3.7. Geographic Anomaly ---
print("  - Generating feature: Geographic Anomaly...")
customer_info = pd.merge(customer_account_linkage_df, customers_df, on='customer_id', how='left')
account_geo_info = pd.merge(accounts_df[['account_id', 'branch_code', 'branch_pin']], customer_info[['account_id', 'customer_pin']], on='account_id', how='left')
account_geo_info['branch_customer_pin_mismatch'] = (account_geo_info['branch_pin'] != account_geo_info['customer_pin']).astype(int)
features_df = pd.merge(features_df, account_geo_info[['account_id', 'branch_customer_pin_mismatch']], on='account_id', how='left')

# --- 3.8. New Account High Value ---
print("  - Generating features: New Account High Value...")
txns_with_open_date = pd.merge(transactions_filtered, accounts_df[['account_id', 'account_opening_date']], on='account_id', how='left')
first_30d_txns = txns_with_open_date[
    txns_with_open_date['transaction_timestamp'] <= (txns_with_open_date['account_opening_date'] + pd.Timedelta(days=30))
]
new_acct_features = first_30d_txns.groupby('account_id').agg(
    total_txn_value_first_30d=('amount', 'sum'),
    txn_count_first_30d=('transaction_id', 'count')
).reset_index()
features_df = pd.merge(features_df, new_acct_features, on='account_id', how='left')

# --- 3.9. Income Mismatch ---
print("  - Generating features: Income Mismatch...")
features_df['credit_turnover_to_balance_ratio'] = features_df['sum_credits'] / (features_df['avg_balance'] + 1e-6) # Add small epsilon to avoid div by zero
features_df['max_txn_to_balance_ratio'] = features_df['max_transaction_amount'] / (features_df['avg_balance'] + 1e-6)

# --- 3.10. Post-Mobile-Change Spike ---
print("  - Generating feature: Post-Mobile-Change Spike...")
if 'last_mobile_update_date' in customers_df.columns:
    customer_mobile_updates = customers_df[['customer_id', 'last_mobile_update_date']].dropna()

    # Correctly link transactions to customers, using the LEAKAGE-PREVENTED transaction set
    transactions_with_customer = pd.merge(transactions_filtered, customer_account_linkage_df, on='account_id', how='left')
    transactions_with_customer = pd.merge(transactions_with_customer, customer_mobile_updates, on='customer_id', how='left')

    def calculate_post_mobile_spike(group):
        # Check if there's any valid mobile update date in the group
        if group['last_mobile_update_date'].isnull().all():
            return pd.Series({'post_mobile_change_txn_count': 0, 'post_mobile_change_txn_value': 0})

        # Use the first valid update date (assuming one update per customer for simplicity)
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
    print("    - Skipping: 'last_mobile_update_date' column not found. Creating dummy features.")
    features_df['post_mobile_change_txn_count'] = 0
    features_df['post_mobile_change_txn_value'] = 0

# --- 3.11. Round Amount Patterns ---
print("  - Generating features: Round Amount Patterns...")
round_amount_thresholds = [1000, 5000, 10000, 50000]
total_txns_series = features_df.set_index('account_id')['total_txn_count'] # For efficient division
for threshold in round_amount_thresholds:
    col_name = f'round_amount_ratio_{threshold}'
    round_counts = transactions_filtered[
        (transactions_filtered['amount'] > 0) & (transactions_filtered['amount'] % threshold == 0)
    ].groupby('account_id').size()
    
    # Map counts and calculate ratio
    features_df[col_name] = features_df['account_id'].map(round_counts) / features_df['account_id'].map(total_txns_series)

# --- 3.12. Salary Cycle Exploitation ---
print("  - Generating feature: Salary Cycle Exploitation...")
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

# --- 3.13. Branch-Level Collusion (Mule Rate per Branch) ---
print("  - Generating feature: Branch-Level Collusion...")
# This feature should ideally be calculated on the training set only to avoid leakage.
# For this pipeline, we'll calculate it based on the provided labels.
branch_mule_rates = accounts_with_labels.groupby('branch_code')['is_mule'].mean().rename('branch_mule_rate').reset_index()
features_df = pd.merge(features_df, branch_mule_rates, on='branch_code', how='left', suffixes=('', '_y'))

# --- 4. Final Cleaning and Output ---
print("4. Final cleaning and output...")

# Fill any remaining NaNs (e.g., for accounts with no transactions for certain types)
features_df.fillna(0, inplace=True)

# Drop intermediate columns that are not features themselves
features_to_drop = [
    'total_txn_count', 
    'credit_txn_count',
    'debit_txn_count',
    'structured_txn_count', 
    'sum_credits', 
    'sum_debits', 
    'avg_transaction_amount',
    'max_transaction_amount',
    'unique_counterparties', 
    'unique_credit_counterparties', 
    'unique_debit_counterparties', 
    'branch_code', 
    'branch_pin', 
    'customer_pin'
]
features_df.drop(columns=features_to_drop, errors='ignore', inplace=True)

# One-hot encode categorical features
features_df = pd.get_dummies(features_df, columns=['account_status', 'product_family', 'rural_branch', 'kyc_compliant', 'nomination_flag'], drop_first=True)

# Ensure 'is_mule' is the last column for convenience in modeling
if 'is_mule' in features_df.columns:
    is_mule_col = features_df.pop('is_mule')
    features_df['is_mule'] = is_mule_col

print(f"[OK] Feature engineering complete. Final DataFrame shape: {features_df.shape}")
print("Sample of generated features:")
print(features_df.head())

# Save the final feature set
features_df.to_csv("engineered_features.csv", index=False)
print("\n[OK] Engineered features saved to 'engineered_features.csv'")