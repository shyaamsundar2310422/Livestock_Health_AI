import pandas as pd
from pathlib import Path

p = Path("dataset_dmi_cownflow/CowNflow_6_Cow_measurements.xlsx")
df = pd.read_excel(p)
df.columns = df.columns.str.strip()

target = "4.-DM-intake-(kg/day)"
cow_id = "1.-Cow-ID"

print("ROWS:", len(df))
print("DMI missing:", df[target].isna().sum())
print("UNIQUE COW IDs:", df[cow_id].nunique())
print("\nDMI SUMMARY:")
print(df[target].describe().to_string())

candidates = [
    "2.-Cow age-(month)",
    "2.-Body weight-(kg)",
    "2.-Physiological status-",
    "2.-Lactation week",
    "2.-Gestation status",
    "2.-Gestation week",
    "3.-Diet type",
    "4.-Diet DM concentration-(g/kg)",
    "4.-Diet CP concentration-(g/kg DM)",
    "4.-Diet NDF concentration-(g/kg DM)"
]

print("\nCANDIDATE FEATURES:")
for col in candidates:
    if col in df.columns:
        print(f"{col}: missing={df[col].isna().sum()}, unique={df[col].nunique()}")
    else:
        print("NOT FOUND:", col)

print("\nFIRST 5 ROWS: CANDIDATE FEATURES + TARGET")
available = [c for c in [cow_id] + candidates + [target] if c in df.columns]
print(df[available].head().to_string(index=False))
