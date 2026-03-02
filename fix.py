# quick_diag.py
import os, pandas as pd, numpy as np

FEATS_PARQUET = os.path.join("data","processed","features","features.parquet")
FEATS_CSV     = os.path.join("data","processed","features","features.csv")

if os.path.exists(FEATS_PARQUET):
    df = pd.read_parquet(FEATS_PARQUET)
else:
    df = pd.read_csv(FEATS_CSV)

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Subjects (sid) unique:", df["sid"].astype(str).nunique())
print(df.groupby("sid").size().sort_values(ascending=False).head(10))
print("Label counts:", df["y"].value_counts(dropna=False))
