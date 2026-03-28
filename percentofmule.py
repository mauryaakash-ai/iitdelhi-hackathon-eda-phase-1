import pandas as pd

customers = pd.read_csv("customers.csv")
accounts = pd.read_csv("accounts.csv")
linkage = pd.read_csv("customer_account_linkage.csv")
products = pd.read_csv("product_details.csv")
labels = pd.read_csv("train_labels.csv")
test = pd.read_csv("test_accounts.csv")


transactions = pd.concat(
    [pd.read_csv(f"transactions_part_{i}.csv") for i in range(6)],
    ignore_index=True
)
for name, df in [("customers", customers), ("accounts", accounts),
                 ("linkage", linkage), ("products", products),
                 ("labels", labels), ("transactions", transactions)]:
    print(f"\n{'='*40}")
    print(f"{name}: {df.shape}")
    print(df.dtypes)
    print(f"\nMissing values:\n{df.isnull().sum()}")
print(labels["is_mule"].value_counts())
print(f"Mule rate: {labels['is_mule'].mean():.4f}")