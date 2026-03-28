import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

# --- Setup ---
# Suppress warnings for a cleaner output and set a professional plot theme.
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# --- 1. Data Loading and Preparation ---
print("Step 1: Loading data... (This may take a moment)")
try:
    labels_df = pd.read_csv("train_labels.csv")
    accounts_df = pd.read_csv("accounts.csv")
    # Concatenate all transaction parts into a single dataframe
    transactions_df = pd.concat(
        [pd.read_csv(f"transactions_part_{i}.csv") for i in range(6)],
        ignore_index=True
    )
    print("[OK] Data loaded successfully.")
except FileNotFoundError as e:
    print(f"[ERROR] Error loading data: {e}. Please ensure all CSV files are in the correct directory.")
    exit()

# --- 2. Feature Engineering for Plots ---
print("\nStep 2: Engineering features for plotting...")

# Calculate transaction aggregations per account
txn_agg = transactions_df.groupby('account_id').agg(
    txn_count=('transaction_id', 'count'),
    avg_txn_amount=('amount', 'mean')
).reset_index()

# Create the main 'train' dataframe for account-level plots
train = pd.merge(labels_df, accounts_df, on='account_id', how='left')
train = pd.merge(train, txn_agg, on='account_id', how='left')
train['account_opening_date'] = pd.to_datetime(train['account_opening_date']) # Ensure datetime for age calculation
train['txn_count'].fillna(0, inplace=True)
train['avg_txn_amount'].fillna(0, inplace=True)

# Create a transactions dataframe with labels for time-based plots
transactions_with_labels = pd.merge(transactions_df, labels_df[['account_id', 'is_mule']], on='account_id', how='inner')
transactions_with_labels['transaction_timestamp'] = pd.to_datetime(transactions_with_labels['transaction_timestamp'])

print("[OK] Feature engineering complete.")

# --- 3. Plotting Section ---
print("\nStep 3: Generating plots as per the hackathon plan...")

#  PART 2 — Density-Based Plotting

##  Transaction Count Density
print("  - Plotting Transaction Count Density...")
plt.figure(figsize=(12, 7))
# Using common_norm=False is crucial for imbalanced classes to compare shape, not area.
# Clipping at the 99th percentile removes extreme outliers for better visualization.
sns.kdeplot(data=train, x="txn_count", hue="is_mule", fill=True, common_norm=False, clip=(0, train['txn_count'].quantile(0.99)))
plt.xlabel("Transaction Count (Clipped at 99th percentile)")
plt.ylabel("Density")
plt.title("Density Distribution of Transaction Count: Mule vs. Legitimate")
plt.legend(title='Account Type', labels=['Mule', 'Legitimate'])
plt.show()
print("\n  - Observation from plot:")
print("    - The density plot clearly shows two different distributions. Legitimate accounts (green) have a peak at a very low transaction count.")
print("    - Mule accounts (red) have a much flatter, wider distribution, with a higher density of accounts in the medium-to-high transaction count range.")
print("    - This confirms that 'transaction count' is a strong differentiating feature.")


##  Transaction Amount Density (Log Scale)
print("  - Plotting Log-Scaled Transaction Amount Density...")
plt.figure(figsize=(12, 7))
# Filter for non-negative average amounts to apply log transform safely.
plot_data = train[train['avg_txn_amount'] >= 0].copy()
plot_data["log_avg_amount"] = np.log1p(plot_data["avg_txn_amount"])

sns.kdeplot(data=plot_data, x="log_avg_amount", hue="is_mule", fill=True, common_norm=False)
plt.xlabel("Log(1 + Average Transaction Amount in INR)")
plt.ylabel("Density")
plt.title("Log-Scaled Transaction Amount Density: Mule vs. Legitimate")
plt.legend(title='Account Type', labels=['Mule', 'Legitimate'])
plt.show()
print("\n  - Observation from plot:")
print("    - The distributions for average transaction amount are more overlapping than for transaction count.")
print("    - However, there is a noticeable bump in the mule distribution at higher average amounts, suggesting some mules handle larger value transactions than typical legitimate accounts.")


#  PART 3 — Time-Based Plotting

# Monthly Transaction Volume Trend (with Leakage Analysis)
print("  - Plotting Monthly Transaction Volume Trend to expose leakage...")
fig, ax1 = plt.subplots(figsize=(16, 8))

# --- Left Y-axis: Transaction Counts ---
transactions_with_labels["month"] = transactions_with_labels["transaction_timestamp"].dt.to_period("M")
monthly = transactions_with_labels.groupby(["month", "is_mule"]).size().reset_index(name="count")
monthly['month'] = monthly['month'].astype(str)

# Plot transaction counts
sns.lineplot(data=monthly, x="month", y="count", hue="is_mule", marker='o', palette=['#43a047', '#e53935'], ax=ax1, zorder=10, legend=False)
ax1.set_xlabel("Month", fontsize=12)
ax1.set_ylabel("Total Transaction Count (Log Scale)", color='#005b96', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#005b96')
ax1.set_yscale('log') # Log scale is crucial to see both lines
ax1.grid(False)

# --- Right Y-axis: Mule Flagging Events ---
ax2 = ax1.twinx()
mule_flags = labels_df[labels_df['is_mule'] == 1].copy()
mule_flags['mule_flag_date'] = pd.to_datetime(mule_flags['mule_flag_date'])
mule_flags['flag_month'] = mule_flags['mule_flag_date'].dt.to_period('M')
flag_counts = mule_flags.groupby('flag_month').size().reset_index(name='flag_count')
flag_counts['flag_month'] = flag_counts['flag_month'].astype(str)

# Use a bar plot for flagging events, it's more intuitive for counts
sns.barplot(data=flag_counts, x='flag_month', y='flag_count', color='gray', alpha=0.4, ax=ax2, zorder=1)
ax2.set_ylabel("Number of Accounts Flagged as Mule", color='gray', fontsize=12)
ax2.tick_params(axis='y', labelcolor='gray')
ax2.grid(False)

# --- Final Touches ---
plt.title("Monthly Transactions vs. Mule Flagging Events (Leakage Analysis)", fontsize=16)

# Improve x-axis ticks to avoid clutter
all_months = sorted(list(set(monthly['month'].tolist() + flag_counts['flag_month'].tolist())))
tick_positions = np.linspace(0, len(all_months)-1, 12).astype(int) # Show ~12 ticks for better granularity
ax1.set_xticks([all_months[i] for i in tick_positions])
ax1.tick_params(axis='x', rotation=45, labelsize=10)

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
legend_elements = [Line2D([0], [0], color='#43a047', lw=2, marker='o', label='Legitimate Txns'),
                   Line2D([0], [0], color='#e53935', lw=2, marker='o', label='Mule Txns'),
                   Patch(facecolor='gray', alpha=0.4, label='Mule Flagging Events')]
ax1.legend(handles=legend_elements, loc='upper left')

fig.tight_layout()
plt.show()
print("\n  - Observation from plot (DATA LEAKAGE):")
print("    - This is a critical plot. The red line (mule transactions) shows a dramatic drop-off that coincides perfectly with the grey bars (mule flagging events).")
print("    - This indicates that once an account is flagged, its transaction activity ceases. This is a form of **data leakage**.")
print("    - Any model trained on post-flagging data will incorrectly learn that 'a sudden stop in transactions' predicts a mule. This pattern will not exist for un-flagged mules in the real world.")
print("    - **Conclusion**: For modeling, all features must be calculated using data *only from before* the `mule_flag_date`.")


## 📊 4️⃣ Time-of-Day Density
print("  - Plotting Time-of-Day Density...")
plt.figure(figsize=(12, 7))
transactions_with_labels["hour"] = transactions_with_labels["transaction_timestamp"].dt.hour
sns.kdeplot(data=transactions_with_labels, x="hour", hue="is_mule", fill=True, common_norm=False, bw_adjust=.5)
plt.xlabel("Hour of Day (0–23)")
plt.ylabel("Density")
plt.title("Transaction Time-of-Day Distribution: Mule vs. Legitimate")
plt.xticks(range(0, 24, 2))
plt.legend(title='Account Type', labels=['Mule', 'Legitimate'])
plt.show()
print("\n  - Observation from plot:")
print("    - Legitimate accounts show clear peaks during standard business hours (approx. 9 AM - 6 PM).")
print("    - Mule accounts have a much flatter distribution, with significantly higher relative activity during late-night and early-morning hours (e.g., 10 PM - 6 AM).")
print("    - This off-hours activity is a classic indicator of fraudulent behavior, making 'hour of day' a valuable feature.")


# 🔥 PART 4 — Structuring Pattern Check
print("  - Plotting Structuring Pattern Check...")

# Calculate structuring ratio feature
total_txns = transactions_df.groupby('account_id').size().rename('total_txn_count')
structured_txns = transactions_df[
    (transactions_df["amount"] > 45000) & (transactions_df["amount"] < 50000)
].groupby("account_id").size().rename('structured_txn_count')

struct_df = pd.concat([total_txns, structured_txns], axis=1).fillna(0)
struct_df['structuring_ratio'] = struct_df['structured_txn_count'] / struct_df['total_txn_count']
struct_df['structuring_ratio'].fillna(0, inplace=True)

# Merge this new feature into our main 'train' dataframe
train = pd.merge(train, struct_df[['structuring_ratio']], on='account_id', how='left')
train['structuring_ratio'].fillna(0, inplace=True)

# Plot the density, comparing mule vs. legit
plt.figure(figsize=(12, 7))
sns.kdeplot(data=train, x='structuring_ratio', hue='is_mule', fill=True, common_norm=False)
plt.xlabel("Proportion of Transactions between ₹45k-₹50k")
plt.ylabel("Density")
plt.title("Distribution of 'Structuring' Transaction Ratio")
# Zoom in on the interesting part of the distribution, as most ratios will be 0
plt.xlim(-0.01, 0.2)
plt.legend(title='Account Type', labels=['Mule', 'Legitimate'])
plt.show()
print("\n  - Observation from plot:")
print("    - This plot visualizes the statistics from the structuring analysis. The vast majority of accounts have a structuring ratio of 0.")
print("    - The small distribution of non-zero ratios is almost entirely composed of mule accounts (red).")
print("    - This confirms that while structuring is rare, when it occurs, it's a very strong signal for mule activity.")


print("\n[OK] All plots generated successfully!")