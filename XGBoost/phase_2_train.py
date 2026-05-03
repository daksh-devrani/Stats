"""
Fraud Detection - Phase 2: Model Training with XGBoost
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import time
from sklearn.metrics import classification_report, roc_auc_score


# ──────────────────────────────────────────────
# LOAD SPLITS FROM PHASE 1
# ──────────────────────────────────────────────

print("=" * 60)
print("Loading Data Splits from Phase 1")
print("=" * 60)

X_train = pd.read_csv("dataAfterI/X_train.csv")
y_train = pd.read_csv("dataAfterI/y_train.csv").squeeze()   # squeeze → Series not DataFrame
X_val   = pd.read_csv("dataAfterI/X_val.csv")
y_val   = pd.read_csv("dataAfterI/y_val.csv").squeeze()

with open("dataAfterI/scale_pos_weight.txt") as f:
    scale_pos_weight = float(f.read().strip())

print(f"X_train : {X_train.shape}")
print(f"X_val   : {X_val.shape}")
print(f"scale_pos_weight loaded : {scale_pos_weight:.2f}")


# ──────────────────────────────────────────────
# STEP 1: CONVERT TO DMATRIX (XGBoost native format)
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 1: Converting to DMatrix")
print("=" * 60)

# DMatrix is XGBoost's optimised internal data structure
# It's faster and more memory-efficient than plain pandas DataFrames
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval   = xgb.DMatrix(X_val,   label=y_val, enable_categorical=True)

print("✅ DMatrix created for train and validation sets.")


# ──────────────────────────────────────────────
# STEP 2: DEFINE HYPERPARAMETERS
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Setting Hyperparameters")
print("=" * 60)

params = {
    # ── Core ──────────────────────────────────
    'objective'        : 'binary:logistic',  # binary classification
    'eval_metric'      : ['logloss', 'auc'],  # track both during training
    'seed'             : 42,                  # reproducibility

    # ── Imbalance ─────────────────────────────
    'scale_pos_weight' : scale_pos_weight,    # handle fraud rarity

    # ── Tree Structure ─────────────────────────
    'max_depth'        : 8,      # depth of each tree (3-10 is typical)
    'n_estimators'     : 300,    # number of trees to build
    'learning_rate'    : 0.05,   # how much each tree contributes (lower = better but slower)

    # ── Regularisation (prevent overfitting) ──
    'subsample'        : 0.8,    # use 80% of rows per tree
    'colsample_bytree' : 0.8,    # use 80% of features per tree
    'min_child_weight' : 5,      # min samples needed to split a node
    'gamma'            : 0.1,    # min loss reduction to make a split

    # ── Speed ─────────────────────────────────
    'tree_method'      : 'hist', # faster histogram-based training
    'device'           : 'cpu',  # change to 'cuda' if you have a GPU
}

print("Parameters:")
for k, v in params.items():
    print(f"  {k:<22} : {v}")


# ──────────────────────────────────────────────
# STEP 3: TRAIN WITH EARLY STOPPING
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 3: Training XGBoost Model")
print("=" * 60)
print("Early stopping: training will auto-stop if AUC doesn't")
print("improve for 30 consecutive rounds. This prevents overfitting.\n")

evals_result = {}   # stores training logs per round

start_time = time.time()

model = xgb.train(
    params          = params,
    dtrain          = dtrain,
    num_boost_round = 500,             # max trees — early stopping will likely kick in before this
    evals           = [(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds = 30,        # stop if val-auc doesn't improve for 30 rounds
    evals_result    = evals_result,    # capture logs
    verbose_eval    = 50,              # print progress every 50 rounds
)

elapsed = time.time() - start_time

print(f"\n⏱  Training Time  : {elapsed:.1f} seconds")
print(f"🌳 Best Round     : {model.best_iteration}")
print(f"🏆 Best Val AUC   : {model.best_score:.6f}")


# ──────────────────────────────────────────────
# STEP 4: TRAINING CURVE — SPOT OVERFITTING
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Training Curve Analysis")
print("=" * 60)

train_auc = evals_result['train']['auc']
val_auc   = evals_result['val']['auc']

best_round  = model.best_iteration
final_round = len(train_auc) - 1

print(f"Round 0    → Train AUC: {train_auc[0]:.4f}  |  Val AUC: {val_auc[0]:.4f}")
print(f"Round {final_round:<4} → Train AUC: {train_auc[-1]:.4f}  |  Val AUC: {val_auc[-1]:.4f}")
print(f"Best Round  {best_round:<4} → Val AUC : {val_auc[best_round]:.4f}")

gap = train_auc[best_round] - val_auc[best_round]
print(f"\nTrain-Val AUC Gap at best round: {gap:.4f}")

if gap < 0.02:
    print("✅ Gap < 0.02 → Model is NOT overfitting. Generalising well.")
elif gap < 0.05:
    print("⚠️  Gap 0.02–0.05 → Slight overfitting. Consider more regularisation.")
else:
    print("❌ Gap > 0.05 → Overfitting! Reduce max_depth or increase min_child_weight.")


# ──────────────────────────────────────────────
# STEP 5: QUICK VALIDATION PEEK
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Quick Validation Peek (threshold = 0.5)")
print("=" * 60)

# Get raw probabilities from the model
val_probs = model.predict(dval)                  # float between 0 and 1
val_preds = (val_probs >= 0.5).astype(int)       # convert to 0 or 1


print(f"Val AUC-ROC : {roc_auc_score(y_val, val_probs):.4f}")
print(f"\nClassification Report (threshold=0.5):")
print(classification_report(y_val, val_preds, target_names=['Legit', 'Fraud']))

print("NOTE: We'll find the OPTIMAL threshold in Phase 3 (Evaluation).")
print("      0.5 is just a sanity check here.")


# ──────────────────────────────────────────────
# SAVE MODEL + PREDICTIONS
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("Saving Model and Outputs")
print("=" * 60)

# Save the model
model.save_model("./outputAfterPhaseII/xgb_fraud_model.json")
print("✅ Saved: xgb_fraud_model.json")

# Save validation probabilities for Phase 3 (evaluation)
pd.DataFrame({
    'y_true' : y_val.values,
    'y_prob' : val_probs
}).to_csv("./outputAfterPhaseII/val_predictions.csv", index=False)
print("✅ Saved: val_predictions.csv  (for Phase 3 evaluation)")

# Save training logs for plotting
pd.DataFrame({
    'round'     : range(len(train_auc)),
    'train_auc' : train_auc,
    'val_auc'   : val_auc,
    'train_loss': evals_result['train']['logloss'],
    'val_loss'  : evals_result['val']['logloss'],
}).to_csv("./outputAfterPhaseII/training_logs.csv", index=False)
print("✅ Saved: training_logs.csv    (for plotting)")

print("\n🎉 Phase 2 Complete! Ready for Phase 3 → Evaluation & Metrics.")
