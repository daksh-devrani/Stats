import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# STEP 1: LOAD DATA
# ──────────────────────────────────────────────

print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

df = pd.read_csv("./archive/fraud_features_enriched.csv", index_col=0)

print(f"Shape   : {df.shape}")
print(f"Columns : {list(df.columns)}")


# ──────────────────────────────────────────────
# STEP 2: BASIC INSPECTION
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Basic Inspection")
print("=" * 60)

print("\n--- Data Types ---")
print(df.dtypes)
print("\n--- First 3 rows ---")
print(df.head(3).to_string())
print("\n--- Basic Stats ---")
print(df.describe())


# ──────────────────────────────────────────────
# STEP 3: MISSING VALUES
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Missing Values Check")
print("=" * 60)

missing = df.isnull().sum()
print(pd.DataFrame({'Missing Count': missing, 'Missing %': (missing/len(df)*100).round(2)}
      )[missing > 0])

if missing.sum() == 0:
    print("No missing values found!")
else:
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        if df[col].isnull().sum() > 0:
            val = df[col].median()
            df[col].fillna(val, inplace=True)
            print(f"  Filled '{col}' with median: {val:.4f}")
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            val = df[col].mode()[0]
            df[col].fillna(val, inplace=True)
            print(f"  Filled '{col}' with mode: {val}")


# ──────────────────────────────────────────────
# STEP 4: DUPLICATE CHECK
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Duplicate Check")
print("=" * 60)

dups = df.duplicated().sum()
print(f"Duplicate rows      : {dups}")
if dups > 0:
    df.drop_duplicates(inplace=True)
    print(f"Removed {dups} duplicate rows.")

trans_dups = df['trans_num'].duplicated().sum()
print(f"Duplicate trans_num : {trans_dups}")
if trans_dups > 0:
    df.drop_duplicates(subset=['trans_num'], inplace=True)
    print(f"Removed {trans_dups} duplicate transaction IDs.")


# ──────────────────────────────────────────────
# STEP 5: CLASS IMBALANCE CHECK
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Class Imbalance Check")
print("=" * 60)

counts = df['is_fraud'].value_counts()
pct    = df['is_fraud'].value_counts(normalize=True) * 100
print(f"Not Fraud (0): {counts[0]:,}  ({pct[0]:.2f}%)")
print(f"Fraud     (1): {counts[1]:,}  ({pct[1]:.2f}%)")
print(f"Imbalance ratio: {counts[0] // counts[1]}:1")
print("→ Will handle with scale_pos_weight in XGBoost.")


# ──────────────────────────────────────────────
# STEP 6: DROP USELESS COLUMNS
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 6: Dropping Useless Columns")
print("=" * 60)

# trans_num  → unique ID, no signal
# first/last → names, useless
# street     → too granular, lat/long covers it
# unix_time  → redundant with trans_date_trans_time
# city       → too many unique values, state covers region
# zip        → redundant with lat/long
# dob        → will extract age below then drop
# NOTE: distance_km, z-scores, IF score → KEPT (already computed by colleague)

cols_to_drop = ['trans_num', 'first', 'last', 'street', 'unix_time', 'city', 'zip']
cols_to_drop = [c for c in cols_to_drop if c in df.columns]

df.drop(columns=cols_to_drop, inplace=True)
print(f"Dropped  : {cols_to_drop}")
print(f"Remaining: {list(df.columns)}")


# ──────────────────────────────────────────────
# STEP 7: DATETIME PARSING & FEATURE EXTRACTION
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 7: DateTime Feature Extraction")
print("=" * 60)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob']                   = pd.to_datetime(df['dob'])

df['trans_hour']    = df['trans_date_trans_time'].dt.hour
df['trans_day']     = df['trans_date_trans_time'].dt.day
df['trans_month']   = df['trans_date_trans_time'].dt.month
df['trans_weekday'] = df['trans_date_trans_time'].dt.weekday
df['is_weekend']    = (df['trans_weekday'] >= 5).astype(int)
df['is_night']      = ((df['trans_hour'] >= 22) | (df['trans_hour'] <= 5)).astype(int)

reference_date = df['trans_date_trans_time'].max()
df['age'] = ((reference_date - df['dob']).dt.days / 365.25).astype(int)

# Sort by time BEFORE saving — phase_1 split depends on this order
df.sort_values('trans_date_trans_time', inplace=True)
df.reset_index(drop=True, inplace=True)

df.drop(columns=['trans_date_trans_time', 'dob'], inplace=True)
print("Extracted : trans_hour, trans_day, trans_month, trans_weekday, is_weekend, is_night, age")
print("Sorted    : by transaction time (required for time-based split in Phase 1)")
print("Dropped   : trans_date_trans_time, dob")


# ──────────────────────────────────────────────
# STEP 8: DISTANCE FEATURE CHECK
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 8: Distance Feature")
print("=" * 60)

if 'distance_km' in df.columns:
    print(f"✅ distance_km already present:")
    print(df['distance_km'].describe().round(2))
else:
    print("⚠️  distance_km missing — computing now...")
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))
    df['distance_km'] = haversine(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    print("✅ Computed: distance_km")


# ──────────────────────────────────────────────
# STEP 9: AMOUNT LOG TRANSFORM
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 9: Amount Log Transform")
print("=" * 60)

Q1, Q3 = df['amt'].quantile(0.25), df['amt'].quantile(0.75)
IQR    = Q3 - Q1
n_out  = ((df['amt'] < Q1 - 3*IQR) | (df['amt'] > Q3 + 3*IQR)).sum()
print(f"Outliers in amt : {n_out:,} ({n_out/len(df)*100:.2f}%) — keeping them, XGBoost handles it")

df['amt_log'] = np.log1p(df['amt'])
print("Created: amt_log")


# ──────────────────────────────────────────────
# STEP 10: ENCODE CATEGORICAL COLUMNS
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 10: Encoding Categorical Columns")
print("=" * 60)

# Gender → binary
df['gender'] = df['gender'].map({'M': 1, 'F': 0})
print("Encoded: gender (M=1, F=0)")

# Category → label encode
category_map = {cat: i for i, cat in enumerate(df['category'].unique())}
df['category'] = df['category'].map(category_map).fillna(-1).astype(int)
print(f"Encoded: category ({len(category_map)} unique values)")

# State → label encode
state_map = {s: i for i, s in enumerate(df['state'].unique())}
df['state'] = df['state'].map(state_map).fillna(-1).astype(int)
print(f"Encoded: state ({len(state_map)} unique values)")

# Merchant & Job → frequency encode
for col in ['merchant', 'job']:
    freq_map = df[col].value_counts().to_dict()
    df[col + '_freq'] = df[col].map(freq_map).fillna(0).astype(int)
    df.drop(columns=[col], inplace=True)
    print(f"Frequency encoded: {col} → {col}_freq")

# cc_num → frequency encode
freq_map = df['cc_num'].value_counts().to_dict()
df['cc_num_freq'] = df['cc_num'].map(freq_map).fillna(0).astype(int)
df.drop(columns=['cc_num'], inplace=True)
print("Frequency encoded: cc_num → cc_num_freq")


# ──────────────────────────────────────────────
# STEP 11: FINAL SANITY CHECK
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 11: Final Sanity Check")
print("=" * 60)

print(f"Final shape    : {df.shape}")
print(f"Missing values : {df.isnull().sum().sum()}")

str_cols = df.select_dtypes(include='object').columns.tolist()
if str_cols:
    print(f"⚠️  String columns still present: {str_cols}")
else:
    print("✅ No string columns — all numeric, safe for XGBoost")

print(f"\nFinal columns:\n{list(df.columns)}")


# ──────────────────────────────────────────────
# STEP 12: SAVE
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 12: Saving Cleaned Data")
print("=" * 60)

import os
os.makedirs("./dataAfterCleaning", exist_ok=True)
df.to_csv("./dataAfterCleaning/clean_data.csv", index=False)

print(f"✅ Saved: ./datAfterCleaning/clean_data.csv  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
print("\n🎉 Cleaning Complete! → Run phase_1_split.py next.")
