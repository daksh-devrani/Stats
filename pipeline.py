import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from scipy.spatial import distance
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, precision_recall_curve, roc_curve, average_precision_score

MODELS_DIR = "models"
XGBOOST_MODEL_PATH = "XGBoost/outputAfterPhaseII/xgb_fraud_model.json"
OPTIMAL_THRESHOLD_PATH = "XGBoost/outputAfterPhaseIII/optimal_threshold.txt"

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def load_models():
    global_stats = joblib.load(os.path.join(MODELS_DIR, "global_stats.joblib"))
    user_stats = joblib.load(os.path.join(MODELS_DIR, "user_stats.joblib"))
    iso_forest = joblib.load(os.path.join(MODELS_DIR, "isolation_forest.joblib"))
    if_features = joblib.load(os.path.join(MODELS_DIR, "if_features.joblib"))
    encodings = joblib.load(os.path.join(MODELS_DIR, "encodings.joblib"))

    xgb_model = xgb.Booster()
    xgb_model.load_model(XGBOOST_MODEL_PATH)

    with open(OPTIMAL_THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())

    return {
        "global_stats": global_stats,
        "user_stats": user_stats,
        "iso_forest": iso_forest,
        "if_features": if_features,
        "xgb_model": xgb_model,
        "threshold": threshold,
        "encodings": encodings
    }

def stage1_classical_stats(df, global_stats, user_stats):
    gs = global_stats

    df['global_amt_zscore'] = (df['amt'] - gs['mean_amt']) / gs['std_amt']

    df['distance_km'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    ).astype('float32')
    df['global_dist_zscore'] = (df['distance_km'] - gs['mean_dist']) / gs['std_dist']

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['date'] = df['trans_date_trans_time'].dt.date
    daily_counts = df.groupby(['cc_num', 'date']).size().reset_index(name='daily_txn_count')
    daily_counts['daily_txn_count'] = daily_counts['daily_txn_count'].astype('int16')
    df = df.merge(daily_counts, on=['cc_num', 'date'], how='left')

    df = df.merge(user_stats, on='cc_num', how='left')

    for col in ['user_mean_amt', 'user_std_amt', 'user_mean_dist', 'user_std_dist',
                'user_mean_velocity', 'user_std_velocity']:
        if col in df.columns:
            df[col] = df[col].fillna(1.0 if 'std' in col else 0.0)

    df['user_amt_zscore'] = np.where(
        df['user_std_amt'] == 0, 0,
        (df['amt'] - df['user_mean_amt']) / df['user_std_amt']
    ).astype('float32')
    df['user_dist_zscore'] = np.where(
        df['user_std_dist'] == 0, 0,
        (df['distance_km'] - df['user_mean_dist']) / df['user_std_dist']
    ).astype('float32')
    df['user_velocity_zscore'] = np.where(
        df['user_std_velocity'] == 0, 0,
        (df['daily_txn_count'] - df['user_mean_velocity']) / df['user_std_velocity']
    ).astype('float32')

    df.drop(columns=['user_mean_amt', 'user_std_amt', 'user_mean_dist',
                      'user_std_dist', 'user_mean_velocity', 'user_std_velocity', 'date'],
            inplace=True, errors='ignore')

    x = df[['amt', 'distance_km', 'daily_txn_count']].values
    mahal = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        mahal[i] = distance.mahalanobis(x[i], gs['mahala_mu'], gs['mahala_inv_cov'])
    df['mahalanobis_dist'] = mahal.astype('float32')

    df['classical_outlier_flag'] = (
        (df['global_amt_zscore'].abs() > 3) |
        (df['global_dist_zscore'].abs() > 3) |
        (df['user_amt_zscore'].abs() > 3) |
        (df['mahalanobis_dist'] > gs['mahala_99_pct'])
    ).astype('int8')

    return df

def stage2_isolation_forest(df, iso_forest, if_features):
    X_if = df[if_features].fillna(0)
    df['if_anomaly_score'] = iso_forest.decision_function(X_if).astype('float32')
    return df

def stage3_xgboost_predict(df, xgb_model, threshold, encodings):
    xgb_features = xgb_model.feature_names
    
    df_xgb = df.copy()
    df_xgb['trans_date_trans_time'] = pd.to_datetime(df_xgb['trans_date_trans_time'])
    df_xgb['dob'] = pd.to_datetime(df_xgb['dob'])
    df_xgb['trans_hour'] = df_xgb['trans_date_trans_time'].dt.hour
    df_xgb['trans_day'] = df_xgb['trans_date_trans_time'].dt.day
    df_xgb['trans_month'] = df_xgb['trans_date_trans_time'].dt.month
    df_xgb['trans_weekday'] = df_xgb['trans_date_trans_time'].dt.weekday
    df_xgb['is_weekend'] = (df_xgb['trans_weekday'] >= 5).astype(int)
    df_xgb['is_night'] = ((df_xgb['trans_hour'] >= 22) | (df_xgb['trans_hour'] <= 5)).astype(int)
    reference_date = df_xgb['trans_date_trans_time'].max()
    df_xgb['age'] = ((reference_date - df_xgb['dob']).dt.days / 365.25).astype(int)

    df_xgb['gender'] = df_xgb['gender'].map({'M': 1, 'F': 0}).fillna(0).astype(int)
    
    # Use saved label encodings
    for col in ['category', 'state']:
        if col in df_xgb.columns:
            mapping = encodings.get(col, {})
            df_xgb[col] = df_xgb[col].map(mapping).fillna(-1).astype(int)

    # Use saved frequency maps
    for col in ['merchant', 'job', 'cc_num']:
        if col in df_xgb.columns:
            freq_map = encodings.get(col + '_freq', {})
            df_xgb[col + '_freq'] = df_xgb[col].map(freq_map).fillna(0).astype(int)

    df_xgb['amt_log'] = np.log1p(df_xgb['amt'])
    
    # --- HYPER-HYBRID INTERACTION ---
    df_xgb['anomaly_weighted_amt'] = df_xgb['amt_log'] * (0.3 - df_xgb['if_anomaly_score'])

    drop_cols = ['trans_date_trans_time', 'dob', 'trans_num', 'first', 'last',
                 'street', 'unix_time', 'city', 'zip', 'merchant', 'job', 'cc_num', 'global_amt_zscore']
    df_xgb.drop(columns=[c for c in drop_cols if c in df_xgb.columns], inplace=True)

    if 'is_fraud' in df_xgb.columns:
        df_xgb.drop(columns=['is_fraud'], inplace=True)

    # Ensure all required features are present and in correct order
    X_input = df_xgb[xgb_features]

    dmatrix = xgb.DMatrix(X_input)
    fraud_probability = xgb_model.predict(dmatrix)

    df['fraud_probability'] = fraud_probability
    df['fraud_prediction'] = (fraud_probability >= threshold).astype(int)

    return df

def run_pipeline(input_path, output_path="pipeline_results.csv"):
    print("=" * 60)
    print("FRAUD DETECTION PIPELINE")
    print("Raw Data → Classical Stats → Isolation Forest → XGBoost")
    print("=" * 60)

    print("\nLoading models...")
    models = load_models()
    print(f"  Optimal threshold: {models['threshold']:.4f}")

    print(f"\nLoading transactions from {input_path}...")
    dtypes = {
        'amt': 'float32', 'lat': 'float32', 'long': 'float32',
        'merch_lat': 'float32', 'merch_long': 'float32',
        'city_pop': 'int32',
    }
    df = pd.read_csv(input_path, dtype=dtypes)
    print(f"  Loaded {len(df):,} transactions")

    print("\n[Stage 1] Classical Statistical Analysis...")
    df = stage1_classical_stats(df, models['global_stats'], models['user_stats'])
    n_classical = df['classical_outlier_flag'].sum()
    print(f"  Classical outliers flagged: {n_classical:,} ({100*n_classical/len(df):.2f}%)")

    print("\n[Stage 2] Isolation Forest Scoring...")
    df = stage2_isolation_forest(df, models['iso_forest'], models['if_features'])
    print(f"  IF anomaly score range: [{df['if_anomaly_score'].min():.4f}, {df['if_anomaly_score'].max():.4f}]")

    print("\n[Stage 3] XGBoost Prediction...")
    df = stage3_xgboost_predict(df, models['xgb_model'], models['threshold'], models['encodings'])
    n_fraud = df['fraud_prediction'].sum()
    print(f"  Predicted fraudulent: {n_fraud:,} ({100*n_fraud/len(df):.2f}%)")

    if 'is_fraud' in df.columns:
        actual_fraud = df['is_fraud'].sum()
        tp = ((df['fraud_prediction'] == 1) & (df['is_fraud'] == 1)).sum()
        fp = ((df['fraud_prediction'] == 1) & (df['is_fraud'] == 0)).sum()
        fn = ((df['fraud_prediction'] == 0) & (df['is_fraud'] == 1)).sum()
        tn = ((df['fraud_prediction'] == 0) & (df['is_fraud'] == 0)).sum()
        
        precision = precision_score(df['is_fraud'], df['fraud_prediction'])
        recall = recall_score(df['is_fraud'], df['fraud_prediction'])
        f1 = f1_score(df['is_fraud'], df['fraud_prediction'])
        auc_roc = roc_auc_score(df['is_fraud'], df['fraud_probability'])
        ap = average_precision_score(df['is_fraud'], df['fraud_probability'])

        print(f"\n  Ground Truth Available:")
        print(f"    Actual fraud: {actual_fraud:,}")
        print(f"    TP: {tp:,} | FP: {fp:,} | FN: {fn:,} | TN: {tn:,}")
        print(f"    Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        print(f"    AUC-ROC: {auc_roc:.4f} | Avg Precision: {ap:.4f}")

        # Export Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(df['is_fraud'], df['fraud_prediction'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Legit', 'Fraud'], 
                    yticklabels=['Legit', 'Fraud'])
        plt.title('Confusion Matrix - Hybrid Pipeline')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('pipeline_confusion_matrix.png')
        plt.close() # Close to free memory
        print(f"\n  ✅ Saved: pipeline_confusion_matrix.png")

        # --- NEW: ROC Curve ---
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(df['is_fraud'], df['fraud_probability'])
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_roc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.savefig('pipeline_roc_curve.png')
        plt.close()
        print(f"  ✅ Saved: pipeline_roc_curve.png")

        # --- NEW: Precision-Recall Curve ---
        plt.figure(figsize=(8, 6))
        prec, rec, _ = precision_recall_curve(df['is_fraud'], df['fraud_probability'])
        plt.plot(rec, prec, color='blue', lw=2, label=f'PR curve (AP = {ap:.4f})')
        # Mark current threshold point
        plt.scatter(recall, precision, color='red', s=100, label=f'Current Threshold ({models["threshold"]:.4f})', zorder=5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        plt.savefig('pipeline_pr_curve.png')
        plt.close()
        print(f"  ✅ Saved: pipeline_pr_curve.png")

        # Export Metrics Report
        with open('pipeline_evaluation_report.txt', 'w') as f:
            f.write("FRAUD DETECTION PIPELINE EVALUATION REPORT\n")
            f.write("="*42 + "\n")
            f.write(f"Dataset: {input_path}\n")
            f.write(f"Total Transactions: {len(df):,}\n")
            f.write(f"Fraud Prevalence: {100*actual_fraud/len(df):.4f}%\n\n")
            f.write("METRICS\n")
            f.write("-" * 7 + "\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall   : {recall:.4f}\n")
            f.write(f"F1 Score : {f1:.4f}\n")
            f.write(f"AUC-ROC  : {auc_roc:.4f}\n")
            f.write(f"Avg Prec : {ap:.4f}\n\n")
            f.write("CONFUSION MATRIX\n")
            f.write("-" * 16 + "\n")
            f.write(f"TP: {tp:,} | FP: {fp:,}\n")
            f.write(f"FN: {fn:,} | TN: {tn:,}\n")
            f.write("\nCLASSIFICATION REPORT\n")
            f.write("-" * 21 + "\n")
            f.write(classification_report(df['is_fraud'], df['fraud_prediction'], target_names=['Legit', 'Fraud']))
        
        print(f"  ✅ Saved: pipeline_evaluation_report.txt")

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    print("=" * 60)

    return df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "archive (3)/fraudTest.csv"

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        sys.exit(1)

    run_pipeline(input_file)
