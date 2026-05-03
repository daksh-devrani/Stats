"""
Fraud Detection - Phase 1: Data Splitting
Single CSV → time-based Train / Val / Test split (70 / 15 / 15)
"""

import pandas as pd
import numpy as np
import os

# ──────────────────────────────────────────────
# LOAD CLEANED DATA
# ──────────────────────────────────────────────

print("=" * 60)
print("Loading Cleaned Data")
print("=" * 60)

# Data is already sorted by time from cleaning step
df = pd.read_csv("./dataAfterCleaning/clean_data.csv")

print(f"Shape : {df.shape}")
print(f"Columns: {list(df.columns)}")


# ──────────────────────────────────────────────
# STEP 1: SEPARATE FEATURES AND TARGET
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 1: Separating Features (X) and Target (y)")
print("=" * 60)

TARGET = 'is_fraud'

X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"\nFeatures:\n{list(X.columns)}")


# ──────────────────────────────────────────────
# STEP 2: TIME-BASED SPLIT  →  70 / 15 / 15
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Time-Based Split (70% train / 15% val / 15% test)")
print("=" * 60)

# Data is already sorted by time — just cut at index boundaries
n          = len(df)
train_end  = int(n * 0.70)
val_end    = int(n * 0.85)   # 70% + 15%

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val   = X.iloc[train_end:val_end]
y_val   = y.iloc[train_end:val_end]

X_test  = X.iloc[val_end:]
y_test  = y.iloc[val_end:]

print(f"Train : rows 0         → {train_end-1}       ({X_train.shape[0]:,} rows  |  70%)")
print(f"Val   : rows {train_end} → {val_end-1}  ({X_val.shape[0]:,} rows  |  15%)")
print(f"Test  : rows {val_end} → {n-1}  ({X_test.shape[0]:,} rows  |  15%)")
print("\n✅ Time-based split preserves temporal order — no future leakage.")


# ──────────────────────────────────────────────
# STEP 3: VERIFY CLASS DISTRIBUTION
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Class Distribution Across Splits")
print("=" * 60)

def class_report(name, y):
    total   = len(y)
    n_fraud = y.sum()
    n_legit = total - n_fraud
    print(f"\n  {name}")
    print(f"    Total : {total:,}")
    print(f"    Legit : {n_legit:,}  ({n_legit/total*100:.2f}%)")
    print(f"    Fraud : {n_fraud:,}  ({n_fraud/total*100:.2f}%)")

class_report("Train", y_train)
class_report("Val  ", y_val)
class_report("Test ", y_test)


# ──────────────────────────────────────────────
# STEP 4: CALCULATE scale_pos_weight
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Calculating scale_pos_weight for XGBoost")
print("=" * 60)

# Calculated from TRAIN only — never peek at val/test
n_legit          = (y_train == 0).sum()
n_fraud          = (y_train == 1).sum()
scale_pos_weight = n_legit / n_fraud

print(f"  Legit cases       : {n_legit:,}")
print(f"  Fraud cases       : {n_fraud:,}")
print(f"  scale_pos_weight  : {n_legit:,} / {n_fraud:,} = {scale_pos_weight:.2f}")
print(f"\n  → XGBoost will treat each fraud sample as {scale_pos_weight:.0f}x more important.")


# ──────────────────────────────────────────────
# SAVE SPLITS
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("Saving Splits")
print("=" * 60)

os.makedirs("./dataAfterI", exist_ok=True)

X_train.to_csv("./dataAfterI/X_train.csv", index=False)
y_train.to_csv("./dataAfterI/y_train.csv", index=False)
X_val.to_csv("./dataAfterI/X_val.csv",     index=False)
y_val.to_csv("./dataAfterI/y_val.csv",     index=False)
X_test.to_csv("./dataAfterI/X_test.csv",   index=False)
y_test.to_csv("./dataAfterI/y_test.csv",   index=False)

with open("./dataAfterI/scale_pos_weight.txt", "w") as f:
    f.write(str(scale_pos_weight))

print("✅ Saved: X_train.csv, y_train.csv")
print("✅ Saved: X_val.csv,   y_val.csv")
print("✅ Saved: X_test.csv,  y_test.csv")
print("✅ Saved: scale_pos_weight.txt")
print("\n🎉 Phase 1 Complete! → Run phase_2_train.py next.")
