import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
import gc

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_mahalanobis(df, cols):
    # Calculate global Mahalanobis distance
    x = df[cols].values
    mu = np.mean(x, axis=0)
    cov = np.cov(x.T)
    try:
        inv_covmat = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        inv_covmat = np.linalg.pinv(cov)
    
    # Calculate distance iteratively to save memory
    # Instead of full matrix multiplication which can be memory heavy:
    left = x - mu
    mahal = np.empty(x.shape[0])
    for i in range(x.shape[0]):
        mahal[i] = distance.mahalanobis(x[i], mu, inv_covmat)
    
    return mahal

def process_data(data_path, output_path="fraud_features_enriched.csv", models_dir="models"):
    os.makedirs(models_dir, exist_ok=True)
    print("Loading data...")
    # Load with optimized types to save memory on 8GB RAM
    dtypes = {
        'cc_num': 'int64',
        'amt': 'float32',
        'lat': 'float32',
        'long': 'float32',
        'merch_lat': 'float32',
        'merch_long': 'float32',
        'city_pop': 'int32',
        'is_fraud': 'int8'
    }
    df = pd.read_csv(data_path, dtype=dtypes, parse_dates=['trans_date_trans_time'])
    
    print("Stage 1: Computing Global & Behavioral Stats...")
    # 1. Global Stats
    mean_amt, std_amt = df['amt'].mean(), df['amt'].std()
    df['global_amt_zscore'] = (df['amt'] - mean_amt) / std_amt
    
    # 2. Distance Feature
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long']).astype('float32')
    mean_dist = df['distance_km'].mean()
    std_dist = df['distance_km'].std()
    df['global_dist_zscore'] = (df['distance_km'] - mean_dist) / std_dist
    df['global_amt_zscore'] = (df['amt'] - mean_amt) / std_amt

    # 3. Velocity Feature
    df['date'] = df['trans_date_trans_time'].dt.date
    daily_counts = df.groupby(['cc_num', 'date']).size().reset_index(name='daily_txn_count')
    daily_counts['daily_txn_count'] = daily_counts['daily_txn_count'].astype('int16')
    df = df.merge(daily_counts, on=['cc_num', 'date'], how='left')
    
    # 4. User-Specific Behavioral Stats
    print("Computing user-specific historical baselines...")
    user_stats = df.groupby('cc_num').agg(
        user_mean_amt=('amt', 'mean'),
        user_std_amt=('amt', 'std'),
        user_mean_dist=('distance_km', 'mean'),
        user_std_dist=('distance_km', 'std'),
        user_mean_velocity=('daily_txn_count', 'mean'),
        user_std_velocity=('daily_txn_count', 'std')
    ).reset_index()
    
    # Fill NaN std devs with 1 to avoid division by zero for users with 1 transaction
    user_stats.fillna({'user_std_amt': 1.0, 'user_std_dist': 1.0, 'user_std_velocity': 1.0}, inplace=True)
    
    df = df.merge(user_stats, on='cc_num', how='left')
    
    df['user_amt_zscore'] = np.where(df['user_std_amt'] == 0, 0, (df['amt'] - df['user_mean_amt']) / df['user_std_amt']).astype('float32')
    df['user_dist_zscore'] = np.where(df['user_std_dist'] == 0, 0, (df['distance_km'] - df['user_mean_dist']) / df['user_std_dist']).astype('float32')
    df['user_velocity_zscore'] = np.where(df['user_std_velocity'] == 0, 0, (df['daily_txn_count'] - df['user_mean_velocity']) / df['user_std_velocity']).astype('float32')

    # Drop intermediate columns to free memory
    df.drop(columns=['user_mean_amt', 'user_std_amt', 'user_mean_dist', 'user_std_dist', 'user_mean_velocity', 'user_std_velocity', 'date'], inplace=True)
    gc.collect()

    print("Computing Multivariate Mahalanobis Distance...")
    mahala_cols = ['amt', 'distance_km', 'daily_txn_count']
    df['mahalanobis_dist'] = compute_mahalanobis(df, mahala_cols).astype('float32')

    x_mahala = df[mahala_cols].values
    mahala_mu = np.mean(x_mahala, axis=0)
    mahala_cov = np.cov(x_mahala.T)
    try:
        mahala_inv_cov = np.linalg.inv(mahala_cov)
    except np.linalg.LinAlgError:
        mahala_inv_cov = np.linalg.pinv(mahala_cov)

    df['classical_outlier_flag'] = (
        (df['global_amt_zscore'].abs() > 3) | 
        (df['global_dist_zscore'].abs() > 3) |
        (df['user_amt_zscore'].abs() > 3) |
        (df['mahalanobis_dist'] > df['mahalanobis_dist'].quantile(0.99))
    ).astype('int8')

    global_stats = {
        'mean_amt': mean_amt,
        'std_amt': std_amt,
        'mean_dist': mean_dist,
        'std_dist': std_dist,
        'mahala_mu': mahala_mu,
        'mahala_inv_cov': mahala_inv_cov,
        'mahala_99_pct': float(df['mahalanobis_dist'].quantile(0.99)),
    }
    
    # Export encodings for pipeline consistency
    # We need to compute these from the training set now
    print("Exporting categorical and frequency maps...")
    encodings = {}
    
    # Label encodings
    for col in ['category', 'state']:
        unique_vals = df[col].unique()
        encodings[col] = {val: i for i, val in enumerate(unique_vals)}
        
    # Frequency encodings
    for col in ['merchant', 'job', 'cc_num']:
        encodings[col + '_freq'] = df[col].value_counts().to_dict()
        
    joblib.dump(encodings, os.path.join(models_dir, 'encodings.joblib'))
    joblib.dump(global_stats, os.path.join(models_dir, 'global_stats.joblib'))
    joblib.dump(user_stats, os.path.join(models_dir, 'user_stats.joblib'))
    print(f"Saved global_stats.joblib, user_stats.joblib, and encodings.joblib to {models_dir}/")

    print("Stage 2: Training Unsupervised Isolation Forest...")
    if_features = [
        'amt', 'distance_km', 'daily_txn_count', 'global_amt_zscore',
        'user_amt_zscore', 'user_dist_zscore', 'user_velocity_zscore',
        'mahalanobis_dist'
    ]
    
    # Subsample data to train IF (100,000 records) to save memory
    train_sample = df.sample(n=min(100000, len(df)), random_state=42)
    X_train = train_sample[if_features].fillna(0)
    
    # Train Isolation Forest
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    
    print("Scoring entire dataset in chunks...")
    # Predict in chunks to prevent memory spikes
    chunk_size = 100000
    if_scores = []
    
    X_full = df[if_features].fillna(0)
    for i in range(0, len(X_full), chunk_size):
        chunk = X_full.iloc[i:i+chunk_size]
        # decision_function gives anomaly score (lower is more abnormal)
        scores = iso_forest.decision_function(chunk)
        if_scores.append(scores)
    
    df['if_anomaly_score'] = np.concatenate(if_scores).astype('float32')

    joblib.dump(iso_forest, os.path.join(models_dir, 'isolation_forest.joblib'))
    joblib.dump(if_features, os.path.join(models_dir, 'if_features.joblib'))
    print(f"Saved isolation_forest.joblib to {models_dir}/")
    
    print(f"Saving enriched dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done! Dataset is ready for XGBoost modeling.")

if __name__ == "__main__":
    train_path = "archive (3)/fraudTrain.csv"
    if os.path.exists(train_path):
        process_data(train_path)
    else:
        print(f"File not found: {train_path}")
