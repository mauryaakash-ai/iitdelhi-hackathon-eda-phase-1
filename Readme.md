
# **Money Mule Detection: Feature Analysis Report**

## **1. Executive Summary**

This report presents the exploratory data analysis (EDA) and feature engineering efforts conducted to identify potential money mule accounts within a large financial dataset.

The primary objective was to uncover behavioral and demographic patterns that differentiate mule accounts from legitimate ones. Through systematic analysis of transaction behavior, account attributes, and customer data, we engineered a robust set of predictive features.

Key insights reveal that mule accounts typically exhibit:

* Rapid movement of funds (quick credit-to-debit cycles)
* High-value transactions in newly created accounts
* Increased activity during non-standard hours

A critical issue of **data leakage** was identified, where transaction records after mule accounts were flagged were included in the dataset. This was resolved by enforcing a strict temporal cutoff—ensuring only pre-flagging data was used for feature engineering.

The final output includes:

* A clean feature dataset (`engineered_features.csv`)
* A comprehensive set of visualizations validating feature effectiveness

This work establishes a strong foundation for building a reliable real-time mule detection system. 

---

## **2. Data Overview**

The analysis utilized the following datasets:

* **accounts.csv** → Account-level static information
* **customers.csv** → Customer demographic data
* **transactions_part_*.csv** → Transaction records (multi-part dataset)
* **train_labels.csv** → Mule classification labels

The dataset is **highly imbalanced**, with mule accounts comprising approximately **2.4%** of total accounts—typical for fraud detection problems.

---

## **3. Critical Finding: Data Leakage**

Initial time-series analysis exposed a major data leakage issue:

* Mule account activity dropped sharply immediately after flagging
* This created an artificial pattern in the dataset

### **Impact**

A model trained on this data would incorrectly learn:

> “A sudden stop in transactions indicates a mule account.”

This pattern would not generalize to real-world scenarios, leading to poor model performance.

### **Mitigation**

* Applied a **temporal filtering strategy**
* For mule accounts → only transactions before `mule_flag_date` were used
* For legitimate accounts → all transactions retained

This correction ensures realistic and unbiased feature generation.

---

## **4. Feature Analysis & Key Findings**

### **4.1 Rapid Pass-Through**

* **Feature:** `pass_through_score`
* **Insight:** Mule accounts frequently move funds out within 24 hours
* **Conclusion:** Strongest behavioral indicator

---

### **4.2 New Account & High-Value Activity**

* **Features:**

  * `account_age_days`
  * `total_txn_value_first_30d`

* **Insight:**

  * Mule accounts are significantly newer
  * High transaction volume early in lifecycle

---

### **4.3 Structuring Behavior**

* **Feature:** `structuring_ratio`

* **Insight:**

  * Transactions clustered just below ₹50,000
  * Rare but highly indicative of mule activity

---

### **4.4 Dormant Account Activation**

* **Features:**

  * `max_dormancy_days`
  * `post_dormancy_burst_volume`

* **Insight:**

  * Long inactivity followed by sudden high activity
  * Strong anomaly pattern

---

### **4.5 Income Mismatch**

* **Features:**

  * `credit_turnover_to_balance_ratio`
  * `max_txn_to_balance_ratio`

* **Insight:**

  * Transaction values disproportionate to account balance
  * Suggests transient fund holding

---

### **4.6 Off-Hours & Salary Cycle Activity**

* **Feature:** `salary_cycle_activity_ratio`

* **Insight:**

  * Irregular transaction timing
  * Higher late-night activity
  * Deviations around salary cycles

---

### **4.7 Geographic & Branch-Level Patterns**

* **Features:**

  * `branch_customer_pin_mismatch`
  * `branch_mule_rate`

* **Insight:**

  * Higher mule concentration in certain branches
  * Address mismatches indicate suspicious behavior

---

### **4.8 Additional Behavioral Signals**

* **Fan-In / Fan-Out:**
  Many counterparties involved in transactions

* **Round Amount Patterns:**
  Frequent use of clean numbers (e.g., ₹10,000)

* **Post-Mobile-Change Spike:**
  Increased activity after mobile number updates

---

## **5. Feature Engineering & Visualization Pipelines**

To ensure reproducibility and consistency:

### **feature_engineering_pipeline.py**

* Handles preprocessing and leakage correction
* Generates all engineered features
* Outputs `engineered_features.csv`

### **feature_visualization_pipeline.py**

* Produces all analytical plots
* Ensures consistency with feature engineering logic
* Saves outputs in `/plots` directory

---

## **6. Conclusion and Next Steps**

### **Most Powerful Features**

**Behavioral:**

* `pass_through_score`
* `account_age_days`
* `structuring_ratio`
* Dormancy-related metrics
* Income mismatch indicators

**Contextual:**

* `branch_mule_rate`
* `branch_customer_pin_mismatch`

---

# 📊 Exploratory Data Analysis (EDA) — Key Observations

## 1. Rapid Pass-Through Behavior

* An iterative scan was performed to identify **credit → debit pairs within 24 hours** covering ≥95% of the credited amount.
* The resulting score distribution indicates:

  * **Mule accounts consistently exhibit higher rapid pass-through scores** compared to legitimate accounts.
* 📌 Supporting visualization: `rapid_pass_through_score_optimized.png`

---

## 2. Clustering Analysis

* Applied **MiniBatchKMeans clustering** on account-level and aggregated transaction features.
* Key findings:

  * **Cluster 0**:

    * Total accounts: **2,098**
    * Mule rate: **8.65%**
    * Median transaction mean: **₹24.5K**
  * Mule accounts are disproportionately concentrated in specific clusters, indicating **behavioral segmentation**.
* 📌 Supporting files:

  * `cluster_scatter.png`
  * `cluster_mule_rate.png`
  * `cluster_summary.csv`

---

## 3. Branch-Level Freeze & Mule Density

* Dot plot analysis highlights branch-level concentration of suspicious activity.
* Key observations:

  * **Branch 4091**:

    * Mule count: **6**
    * Freeze count: **3**
    * Mule rate (labeled): **85.7%**
  * **Branch 2847**:

    * Mule count: **2**
    * Freeze count: **2**
    * Mule rate: **50%**
* These branches show **elevated operational risk** and may require targeted monitoring.
* 📌 Data source: `branch_freeze_mule_stats.csv`
* 📌 Visualization: `branch_freeze_mule_dotplots.png`

---

## 4. Freeze → Next Account Opening (Same Branch)

* Analyzed the time gap between **account freeze and next account opening within the same branch**.

### Overall Statistics:

* Median gap: **167 days**
* 75th percentile: **320 days**
* Maximum: **1,268 days**
* Sample size: **950 freeze events with subsequent openings**

### Branch-Level Highlights:

* **Branch 5672**: Median gap **101 days** (n=3)

* **Branch 4091**: Gap **65 days** (n=1)

* **Branch 3984**: Gap **88 days** (n=1)

* Insights suggest **re-entry into the system after freeze events**, possibly indicating behavioral recurrence.

* 📌 Supporting files:

  * `freeze_gap_branch_bubble.png`
  * `freeze_gap_hist_by_label.png`
  * `freeze_to_open_branch_stats.csv`
  * `freeze_to_open_events.csv`

---

## 5. Frozen Account Alert Reasons

Among labeled frozen accounts, the most frequent alert triggers are:

| Alert Reason                | Count |
| --------------------------- | ----: |
| Routine Investigation       |    40 |
| Rapid Movement of Funds     |    14 |
| Structuring Below Threshold |    11 |
| Unusual Fund Flow Pattern   |    10 |
| Income–Transaction Mismatch |     8 |

* These reasons align strongly with **known mule behavior patterns**.

* 📌 Data source: `frozen_accounts_alert_reasons.csv`

---

## 6. Core Data Audits

* Performed audits on:

  * Customers
  * Accounts
  * Linkages
  * Products
  * Labels

### Key Findings:

* **Class imbalance**:

  * Mule accounts: **263**
  * Legitimate accounts: **23,760**
  * Mule rate: **1.09%**
* **Missingness**:

  * Freeze/unfreeze dates: largely missing
  * IDs and KYC fields: mostly complete

---

## 7. Customer Demographics (Mule vs Legit)

* **Age**:

  * Median: ~**50 years** (both groups)
  * Mules slightly older
* **Tenure**:

  * Median: ~**16 years**
  * Mules show marginally longer tenure

➡️ Conclusion:
Demographics alone are **weak discriminators**.

---

## 8. Product Holdings

* Mule accounts exhibit:

  * Slightly higher **loan exposure**
  * Slightly higher **credit card usage**
  * Higher **median savings balance** (₹1.1K vs ₹0 for legit)

➡️ Suggests **financial activity layering**, not inactivity.

---

## 9. Transaction Channel Analysis (Labeled Accounts)

* Mule activity is heavily concentrated in:

  * **UPI (UPC / UPD)**
  * **IMPS**

### Top Channels (Mule Transactions):

* UPC: **16,855**
* UPD: **15,187**
* IPM: **3,174**

---

## 10. Amount Pattern Signals

* Strong structuring indicators observed:

  * **₹1,000 multiples**:

    * 7,935 mule transactions
  * **₹49K–₹50K band**:

    * 223 mule transactions

➡️ Indicates:

* **Threshold-aware structuring behavior**
* Attempts to avoid regulatory detection limits

---

# Overall Insight

Across multiple dimensions—transaction velocity, clustering behavior, branch-level concentration, and structuring patterns—**mule accounts exhibit distinct, repeatable signatures**.

These signals can be leveraged to:

* Build **risk scoring models**
* Enable **early detection systems**
* Drive **branch-level interventions**





