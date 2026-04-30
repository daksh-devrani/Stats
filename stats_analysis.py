import pandas as pd
import numpy as np
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def perform_classical_stats_analysis(data_path, output_path="outliers_detected.csv"):
    df = pd.read_csv(data_path)
    
    mean_amt = df['amt'].mean()
    std_amt = df['amt'].std()
    df['outlier_zscore'] = ((df['amt'] - mean_amt) / std_amt) > 3
    
    Q1 = df['amt'].quantile(0.25)
    Q3 = df['amt'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df['outlier_iqr'] = df['amt'] > upper_bound
    
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    mean_dist = df['distance_km'].mean()
    std_dist = df['distance_km'].std()
    df['outlier_distance'] = ((df['distance_km'] - mean_dist) / std_dist) > 3

    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['date'] = df['trans_date_trans_time'].dt.date
    daily_counts = df.groupby(['cc_num', 'date']).size().reset_index(name='daily_txn_count')
    df = df.merge(daily_counts, on=['cc_num', 'date'], how='left')
    
    mean_daily = df['daily_txn_count'].mean()
    std_daily = df['daily_txn_count'].std()
    df['outlier_velocity'] = ((df['daily_txn_count'] - mean_daily) / std_daily) > 3
    
    df['is_outlier'] = df['outlier_zscore'] | df['outlier_iqr'] | df['outlier_distance'] | df['outlier_velocity']
    
    outliers = df[df['is_outlier']]
    outliers.to_csv(output_path, index=False)
    
    print(f"Total records processed: {len(df)}")
    print(f"Total statistical outliers detected: {len(outliers)}")
    print(f"Outliers saved to: {output_path}")
    
    return outliers

if __name__ == "__main__":
    train_path = "archive (3)/fraudTrain.csv"
    if os.path.exists(train_path):
        perform_classical_stats_analysis(train_path)
    else:
        print(f"File not found: {train_path}")
