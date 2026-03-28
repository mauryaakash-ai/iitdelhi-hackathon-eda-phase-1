import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Sirf zaroori file 'train_labels.csv' ko load karein
print("Loading train_labels.csv...")
labels_df = pd.read_csv("train_labels.csv")
print("Load complete.")

# Sirf mule accounts (is_mule == 1) ko filter karein
mule_accounts = labels_df[labels_df['is_mule'] == 1].copy()

print("\n" + "="*50)
print("      Mule Account Alert Reason Analysis")
print("="*50)

# 'alert_reason' ke liye value counts aur percentage nikalen
reason_counts = mule_accounts['alert_reason'].value_counts()
reason_percentages = mule_accounts['alert_reason'].value_counts(normalize=True) * 100

analysis_summary = pd.DataFrame({
    'Reason': reason_counts.index,
    'Count': reason_counts.values,
    'Percentage': reason_percentages.values
})

print("\nReasons for flagging mule accounts:\n")
print(analysis_summary.to_string(index=False))

# --- Visualization ---
try:
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Percentage', y='Reason', data=analysis_summary, palette='viridis', hue='Reason', dodge=False)
    plt.title('Distribution of Alert Reasons for Mule Accounts', fontsize=16)
    plt.xlabel('Percentage of Mule Accounts (%)', fontsize=12)
    plt.ylabel('Alert Reason', fontsize=12)
    plt.legend([],[], frameon=False) # Legend ko hatane ke liye
    plt.tight_layout()
    plt.show()
    print("\n[SUCCESS] Reasons ka distribution plot generate ho gaya hai.")

except (ImportError, NameError):
    print("\n[INFO] Matplotlib ya Seaborn install nahi hai. Visualization skip kar rahe hain.")
    print("Plot generate karne ke liye, install karein: pip install matplotlib seaborn")