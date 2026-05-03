"""
Fraud Detection - Phase 3: Model Evaluation
- Confusion Matrix
- Precision, Recall, F1, AUC-ROC
- Optimal Threshold Finding
- Training Curve Plot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# ──────────────────────────────────────────────
# LOAD PREDICTIONS + TRAINING LOGS
# ──────────────────────────────────────────────

print("=" * 60)
print("Loading Phase 2 Outputs")
print("=" * 60)

val_df   = pd.read_csv("outputAfterPhaseII/val_predictions.csv")
logs_df  = pd.read_csv("outputAfterPhaseII/training_logs.csv")

y_true   = val_df['y_true'].values
y_prob   = val_df['y_prob'].values

print(f"Validation samples : {len(y_true):,}")
print(f"Fraud cases        : {y_true.sum():,}")
print(f"Training rounds    : {len(logs_df)}")


# ──────────────────────────────────────────────
# STEP 1: CORE METRICS AT THRESHOLD 0.5
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 1: Core Metrics at Default Threshold (0.5)")
print("=" * 60)

y_pred_05 = (y_prob >= 0.5).astype(int)

auc      = roc_auc_score(y_true, y_prob)
f1       = f1_score(y_true, y_pred_05)
prec     = precision_score(y_true, y_pred_05)
rec      = recall_score(y_true, y_pred_05)
avg_prec = average_precision_score(y_true, y_prob)

print(f"  AUC-ROC            : {auc:.4f}   ← main metric (1.0 = perfect)")
print(f"  Avg Precision (AP) : {avg_prec:.4f}   ← area under PR curve")
print(f"  F1 Score           : {f1:.4f}   ← balance of precision & recall")
print(f"  Precision          : {prec:.4f}   ← of flagged frauds, how many real?")
print(f"  Recall             : {rec:.4f}   ← of real frauds, how many caught?")


# ──────────────────────────────────────────────
# STEP 2: FIND OPTIMAL THRESHOLD
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Finding Optimal Threshold")
print("=" * 60)

# Strategy: sweep all thresholds and pick the one with best F1
# In fraud detection we care more about catching fraud (recall)
# so we use F1 which balances both precision and recall

precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-9)

best_idx       = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_f1        = f1_scores[best_idx]
best_precision = precisions[best_idx]
best_recall    = recalls[best_idx]

print(f"  Optimal Threshold  : {best_threshold:.4f}")
print(f"  F1 at optimal      : {best_f1:.4f}")
print(f"  Precision          : {best_precision:.4f}")
print(f"  Recall             : {best_recall:.4f}")
print(f"\n  (Default 0.5 gave F1 = {f1:.4f})")
print(f"  Improvement        : +{(best_f1 - f1):.4f} F1 just from threshold tuning!")

# Save optimal threshold for Phase 6
with open("./outputAfterPhaseIII/optimal_threshold.txt", "w") as f:
    f.write(str(best_threshold))
print(f"\n✅ Saved optimal threshold → optimal_threshold.txt")


# ──────────────────────────────────────────────
# STEP 3: FULL REPORT AT OPTIMAL THRESHOLD
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"STEP 3: Full Report at Optimal Threshold ({best_threshold:.4f})")
print("=" * 60)

y_pred_opt = (y_prob >= best_threshold).astype(int)
print(classification_report(y_true, y_pred_opt, target_names=['Legit', 'Fraud'], digits=4))


# ──────────────────────────────────────────────
# STEP 4: CONFUSION MATRIX
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 4: Confusion Matrix")
print("=" * 60)

cm = confusion_matrix(y_true, y_pred_opt)
tn, fp, fn, tp = cm.ravel()

print(f"  True  Negatives (TN) : {tn:,}  ← legit correctly identified")
print(f"  False Positives (FP) : {fp:,}  ← legit wrongly flagged as fraud")
print(f"  False Negatives (FN) : {fn:,}  ← fraud missed  ⚠️  (costly!)")
print(f"  True  Positives (TP) : {tp:,}  ← fraud correctly caught ✅")
print(f"\n  False Negative Rate  : {fn/(fn+tp)*100:.2f}%  (fraud we missed)")
print(f"  False Positive Rate  : {fp/(fp+tn)*100:.2f}%  (legit we blocked)")


# ──────────────────────────────────────────────
# STEP 5: VISUALISATIONS (4 plots)
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 5: Generating Evaluation Plots")
print("=" * 60)

fig = plt.figure(figsize=(18, 14))
fig.suptitle("Fraud Detection Model — Phase 3 Evaluation", fontsize=16, fontweight='bold', y=1.01)
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# ── Plot 1: Training Curve ─────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(logs_df['round'], logs_df['train_auc'], label='Train AUC', color='steelblue',  linewidth=2)
ax1.plot(logs_df['round'], logs_df['val_auc'],   label='Val AUC',   color='darkorange', linewidth=2)
ax1.axvline(logs_df['round'].iloc[-1], color='red', linestyle='--', alpha=0.6, label=f'Best Round')
ax1.set_title("Training Curve (AUC over Rounds)", fontweight='bold')
ax1.set_xlabel("Boosting Round")
ax1.set_ylabel("AUC-ROC")
ax1.legend()
ax1.grid(alpha=0.3)

# ── Plot 2: Confusion Matrix ───────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
labels = np.array([[f"{v}\n({p:.1f}%)" for v, p in zip(row_v, row_p)]
                   for row_v, row_p in zip(cm, cm_pct)])
sns.heatmap(
    cm_pct, annot=labels, fmt='', cmap='Blues',
    xticklabels=['Pred: Legit', 'Pred: Fraud'],
    yticklabels=['True: Legit', 'True: Fraud'],
    ax=ax2, linewidths=1, linecolor='white', cbar=False, annot_kws={"size": 13}
)
ax2.set_title("Confusion Matrix", fontweight='bold')
ax2.set_ylabel("Actual")
ax2.set_xlabel("Predicted")

# ── Plot 3: ROC Curve ──────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
fpr, tpr, _ = roc_curve(y_true, y_prob)
ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {auc:.4f}')
ax3.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random classifier')
ax3.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
ax3.set_title("ROC Curve", fontweight='bold')
ax3.set_xlabel("False Positive Rate (Legit flagged as Fraud)")
ax3.set_ylabel("True Positive Rate (Fraud caught)")
ax3.legend()
ax3.grid(alpha=0.3)

# ── Plot 4: Precision-Recall Curve ────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(recalls[:-1], precisions[:-1], color='steelblue', lw=2, label=f'AP = {avg_prec:.4f}')
ax4.axvline(best_recall,    color='red',   linestyle='--', alpha=0.7, label=f'Optimal threshold ({best_threshold:.2f})')
ax4.axhline(best_precision, color='green', linestyle='--', alpha=0.7)
ax4.scatter([best_recall], [best_precision], color='red', s=100, zorder=5)
ax4.set_title("Precision-Recall Curve", fontweight='bold')
ax4.set_xlabel("Recall (Fraud Caught)")
ax4.set_ylabel("Precision (Flagged = actually Fraud)")
ax4.legend()
ax4.grid(alpha=0.3)

plt.savefig("./outputAfterPhaseIII/phase3_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved: phase3_evaluation.png")


# ──────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 3 SUMMARY")
print("=" * 60)
print(f"  AUC-ROC            : {auc:.4f}")
print(f"  Best F1 Score      : {best_f1:.4f}  (at threshold {best_threshold:.4f})")
print(f"  Precision          : {best_precision:.4f}")
print(f"  Recall             : {best_recall:.4f}")
print(f"  Fraud caught       : {tp:,} / {tp+fn:,}  ({tp/(tp+fn)*100:.1f}%)")
print(f"  False alarms       : {fp:,}")
print("\n🎉 Phase 3 Complete! Ready for Phase 4 → Feature Importance.")