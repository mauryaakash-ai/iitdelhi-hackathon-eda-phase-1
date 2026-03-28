Money Mule Detection: Feature Analysis Report
1. Executive Summary

This report details the exploratory data analysis (EDA) and feature engineering process undertaken to identify potential money mule accounts from a large financial dataset. The primary goal was to uncover behavioral and demographic patterns that distinguish mule accounts from legitimate ones.

Through a systematic analysis of transaction patterns, account characteristics, and customer information, we have successfully engineered a set of powerful predictive features. Key findings indicate that mule accounts often exhibit behaviors such as rapid fund pass-through, high-value activity in newly opened accounts, and a higher proportion of off-hours transactions.

A critical discovery of data leakage was made, where transaction data for mule accounts was available after they were flagged. This issue was addressed by implementing a strict temporal cutoff in our feature engineering pipeline, ensuring that only data prior to the flagging date was used for feature calculation.

The final output is a clean, engineered feature set (engineered_features.csv) and a comprehensive set of visualizations that validate the predictive power of these features. This work lays a strong foundation for building a robust machine learning model for real-time mule detection.
2. Data Overview

The analysis was conducted on a comprehensive dataset comprising:

    accounts.csv: Static account-level information.
    customers.csv: Customer demographic data.
    transactions_part_*.csv: A multi-part dataset containing all transaction records.
    train_labels.csv: Account labels indicating whether an account is a mule (is_mule=1) or legitimate (is_mule=0).
    

The dataset is highly imbalanced, with mule accounts representing a small fraction (~2.4%) of the total, which is typical for fraud detection scenarios.
3. Critical Finding: Data Leakage

Initial time-series analysis revealed a significant data leakage issue. The transaction activity for known mule accounts showed a sharp drop-off that perfectly coincided with the dates they were flagged.

!Monthly Transaction Volume Figure: The red line (mule transactions) drops dramatically as flagging events (grey bars) occur, indicating that post-flagging data (where activity ceases) was included in the original dataset. This is a form of data leakage.

Impact: A model trained on this data would incorrectly learn that "a sudden stop in activity" is a predictor of a mule account. This pattern would not exist for new, un-flagged mules in a real-world scenario, leading to a poorly performing model.

Mitigation: All feature engineering was performed on a temporally filtered transaction set. For any known mule account, only transactions occurring before its mule_flag_date were used. For legitimate accounts, all transactions were retained. This approach is implemented in both feature_engineering_pipeline.py and feature_visualization_pipeline.py.
4. Feature Analysis and Key Findings

The following section details the analysis of various potential mule activity patterns. For each, we describe the engineered feature and present a visualization comparing its distribution between mule and legitimate accounts.
4.1. Rapid Pass-Through

    Hypothesis: Mules quickly move funds in and out of an account.
    Feature: pass_through_score - Counts instances where a credit transaction is followed by a debit of at least 95% of the credit amount within 24 hours.
    Observation: The median and distribution of the pass-through score are significantly higher for mule accounts. This is a very strong indicator.

!Rapid Pass-Through Score
4.2. New Account & High-Value Activity

    Hypothesis: Mules often use newly opened accounts for high-volume or high-value transactions.
    Features:
        account_age_days: The age of the account in days.
        total_txn_value_first_30d: Sum of transaction amounts in the first 30 days.
    Observation: Mule accounts are, on average, significantly younger than legitimate accounts. They also exhibit much higher total transaction values within the first 30 days of opening.

!Account Age !New Account High Value
4.3. Structuring

    Hypothesis: Mules make many transactions just below a reporting threshold (e.g., ₹50,000) to avoid scrutiny.
    Feature: structuring_ratio - The proportion of an account's transactions that fall between ₹45,000 and ₹50,000.
    Observation: While rare overall, a non-zero structuring ratio is almost exclusively observed in mule accounts, making it a powerful, albeit infrequent, signal.

!Structuring Ratio
4.4. Dormant Account Activation

    Hypothesis: Dormant accounts are suddenly activated with a burst of high-volume/high-value activity.
    Features:
        max_dormancy_days: The longest period (in days) between consecutive transactions.
        post_dormancy_burst_volume: Total value of transactions in the 30 days following the longest dormancy period.
    Observation: Mule accounts show a clear tendency for higher transaction volumes immediately following a long period of inactivity compared to legitimate accounts.

!Max Dormancy !Post-Dormancy Burst Volume
4.5. Income Mismatch

    Hypothesis: The total value of credits or the size of the largest transaction is disproportionately high compared to the account's average balance.
    Features:
        credit_turnover_to_balance_ratio: Total credits divided by the average balance.
        max_txn_to_balance_ratio: Maximum single transaction amount divided by the average balance.
    Observation: Mule accounts exhibit significantly higher ratios for both credit turnover and maximum transaction size relative to their average balance, suggesting funds are not held long and are inconsistent with typical account holdings.

!Credit Turnover Ratio !Max Txn to Balance Ratio
4.6. Off-Hours & Salary Cycle Activity

    Hypothesis: Mules conduct transactions at unusual times (e.g., late at night) or exploit typical salary deposit cycles for cover.
    Features:
        salary_cycle_activity_ratio: Proportion of transactions occurring around the start/end of the month (days 25-31 and 1-5).
    Observation: The salary_cycle_activity_ratio shows a slightly different distribution, with mule accounts having a higher density of accounts with either very low or very high activity in this window, deviating from the norm. Analysis of transaction hour (from EDA) also showed mules have higher relative activity during late-night hours.

!Salary Cycle Ratio
4.7. Geographic & Branch-Level Anomalies

    Hypothesis: Mules might be concentrated in specific branches (collusion) or have a mismatch between their registered address and the branch location.
    Features:
        branch_customer_pin_mismatch: A flag indicating if the customer's PIN code differs from the account's branch PIN code.
        branch_mule_rate: The historical average of mule accounts per branch.
    Observation: A PIN code mismatch, while not overwhelmingly common, occurs more frequently among mule accounts. Furthermore, the branch_mule_rate shows that some branches have a significantly higher concentration of mule accounts, suggesting this could be a powerful feature for identifying localized fraud rings.

!PIN Mismatch !Branch Mule Rate
4.8. Other Behavioral Patterns

    Fan-In / Fan-Out: Mule accounts tend to receive funds from many sources and disburse to many destinations. Features like unique_credit_counterparties and unique_debit_counterparties capture this.
    Round Amount Patterns: Mules may use round-number transactions (e.g., exactly ₹10,000). The round_amount_ratio_* features quantify this.
    Post-Mobile-Change Spike: A spike in activity after a mobile number update can be suspicious. Features like post_mobile_change_txn_count were created to monitor this.

5. Feature Engineering & Visualization Pipelines

To ensure a reproducible and robust workflow, two key pipelines were created:

    feature_engineering_pipeline.py: This script automates the entire process of loading raw data, performing preprocessing, applying the critical data leakage fix, calculating all the features described above, and saving the final model-ready dataset as engineered_features.csv.

    feature_visualization_pipeline.py: This script was developed to systematically generate and save all the plots included in this report. It uses the same feature calculation logic as the engineering pipeline to ensure consistency and provides clear, visual evidence of the predictive power of each feature. All plots are saved to the /plots directory.

6. Conclusion and Next Steps

The analysis successfully identified several strong indicators of money mule activity. The most promising features are:

    Behavioral: pass_through_score, account_age_days, structuring_ratio, and metrics related to dormant account activation and income mismatch.
    Contextual: branch_mule_rate and branch_customer_pin_mismatch.

The engineered feature set is now ready for model development. The recommended next steps are:

    Model Training: Split the engineered_features.csv dataset into training and testing sets.
    Model Evaluation: Train several classification models (e.g., Logistic Regression, XGBoost, LightGBM, Random Forest) and evaluate their performance using metrics appropriate for imbalanced datasets, such as the F1-score, Precision-Recall AUC, and confusion matrices.
    Hyperparameter Tuning: Fine-tune the best-performing model to optimize its predictive power.
    Deployment: Prepare the final model for deployment in a real-time transaction monitoring system.


Here’s a **clean, structured, and professional version** of your EDA observations 👇

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





