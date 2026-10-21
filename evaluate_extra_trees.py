import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    f1_score, recall_score, precision_score, cohen_kappa_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import ExtraTreesRegressor

def run_fine_tuned_extra_trees():
    print('--- 1. LOADING & PREPROCESSING PM25.CSV ---')
    df = pd.read_csv('pm25.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Physical outlier filter (sensor errors)
    initial_len = len(df)
    df = df[(df['pm25'] > 0) & (df['pm25'] <= 350)].copy()
    print(f'Grounded {initial_len - len(df)} sensor error outliers. Valid samples: {len(df)}')

    # Sort chronologically
    df = df.sort_values(['point_id', 'date']).reset_index(drop=True)

    # Meteorological Feature Engineering
    rad = np.radians(df['wind_dir_deg'].fillna(180))
    df['wind_u'] = -df['wind_speed_station_m_s'] * np.sin(rad)
    df['wind_v'] = -df['wind_speed_station_m_s'] * np.cos(rad)
    df['ventilation_idx'] = df['wind_speed_station_m_s'] * df['mixing_height_station_m']
    df['stagnation_idx'] = 1.0 / (df['ventilation_idx'] + 1e-3)
    
    # LST Handling (0.0 is missing satellite scan / cloud mask)
    df['lst_missing'] = (df['lst_celsius'] == 0) | (df['lst_celsius'].isna()).astype(int)
    df['lst_imputed'] = df['lst_celsius'].replace(0, np.nan)
    df['lst_imputed'] = df.groupby('point_id')['lst_imputed'].ffill().bfill().fillna(26.0)

    # Impute missing weather
    for col in ['wind_speed_station_m_s', 'wind_u', 'wind_v', 'mixing_height_station_m', 
                'ventilation_idx', 'stagnation_idx', 'lst_imputed']:
        df[col] = df.groupby('point_id')[col].ffill().bfill().fillna(df[col].median())

    # Hotspot features
    for col in ['hotspot_10km_count', 'hotspot_25km_count', 'hotspot_50km_count', 'hotspot_province_count']:
        df[col] = df[col].fillna(0)
    df['hotspot_local_ratio'] = (df['hotspot_10km_count'] + 1) / (df['hotspot_50km_count'] + 1)

    # Calendar & Seasonality
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_burning_season'] = df['month'].isin([1, 2, 3, 4]).astype(int)

    # Spatio-Temporal Autoregressive & Rolling Lags
    for lag in [1, 2, 3, 7]:
        df[f'pm25_lag_{lag}'] = df.groupby('point_id')['pm25'].shift(lag)
        for h_col in ['hotspot_25km_count', 'hotspot_province_count']:
            df[f'{h_col}_lag_{lag}'] = df.groupby('point_id')[h_col].shift(lag)

    df['pm25_roll_mean_3d'] = df.groupby('point_id')['pm25'].shift(1).rolling(3, min_periods=1).mean()
    df['pm25_roll_std_3d'] = df.groupby('point_id')['pm25'].shift(1).rolling(3, min_periods=1).std().fillna(0)
    df['pm25_roll_mean_7d'] = df.groupby('point_id')['pm25'].shift(1).rolling(7, min_periods=1).mean()
    df['pm25_roll_max_7d'] = df.groupby('point_id')['pm25'].shift(1).rolling(7, min_periods=1).max()
    df['hotspot_province_roll_sum_3d'] = df.groupby('point_id')['hotspot_province_count'].shift(1).rolling(3, min_periods=1).sum()
    df['hotspot_province_roll_sum_7d'] = df.groupby('point_id')['hotspot_province_count'].shift(1).rolling(7, min_periods=1).sum()

    # Province-wide spatial background lag
    daily_prov = df.groupby('date')['pm25'].mean().rename('prov_pm25_mean')
    df = df.merge(daily_prov, on='date', how='left')
    df['prov_pm25_mean_lag_1'] = df.groupby('point_id')['prov_pm25_mean'].shift(1)

    df_model = df.dropna().reset_index(drop=True)

    # Split: Train on historical data, Future Backtest on unseen data
    split_date = '2025-06-01'
    train_mask = df_model['date'] < split_date
    test_mask = df_model['date'] >= split_date

    feature_cols = [
        'lat', 'lon',
        'sin_doy', 'cos_doy', 'is_burning_season', 'month', 'dayofweek', 'is_weekend',
        'wind_speed_station_m_s', 'wind_u', 'wind_v',
        'mixing_height_station_m', 'ventilation_idx', 'stagnation_idx',
        'lst_imputed', 'lst_missing',
        'hotspot_10km_count', 'hotspot_25km_count', 'hotspot_50km_count', 'hotspot_province_count',
        'hotspot_local_ratio',
        'hotspot_25km_count_lag_1', 'hotspot_25km_count_lag_2',
        'hotspot_province_count_lag_1', 'hotspot_province_count_lag_2', 'hotspot_province_count_lag_3',
        'hotspot_province_roll_sum_3d', 'hotspot_province_roll_sum_7d',
        'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_7',
        'pm25_roll_mean_3d', 'pm25_roll_std_3d', 'pm25_roll_mean_7d', 'pm25_roll_max_7d',
        'prov_pm25_mean_lag_1'
    ]

    X_train, y_train = df_model.loc[train_mask, feature_cols], df_model.loc[train_mask, 'pm25']
    X_test, y_test = df_model.loc[test_mask, feature_cols], df_model.loc[test_mask, 'pm25']

    train_start = df_model.loc[train_mask, 'date'].min().date()
    train_end = df_model.loc[train_mask, 'date'].max().date()
    test_start = df_model.loc[test_mask, 'date'].min().date()
    test_end = df_model.loc[test_mask, 'date'].max().date()
    print(f'Train Samples: {len(X_train)} ({train_start} to {train_end})')
    print(f'Test Samples:  {len(X_test)} ({test_start} to {test_end})')

    # Train Fine-Tuned Extra Trees Regressor
    print('\n--- 2. TRAINING FINE-TUNED EXTRA TREES REGRESSOR ---')
    model = ExtraTreesRegressor(
        n_estimators=250,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = np.clip(model.predict(X_test), 0, None)

    # 1. Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f'Continuous Regression Metrics:')
    print(f'  RMSE: {rmse:.3f} ug/m3')
    print(f'  MAE:  {mae:.3f} ug/m3')
    print(f'  R2:   {r2:.4f} ({r2*100:.2f}%)')

    # 2. Binary Unhealthy Alert (Standard Threshold >= 37.5 ug/m3)
    y_true_b37 = (y_test >= 37.5).astype(int)
    y_pred_b37 = (preds >= 37.5).astype(int)

    rec_37 = recall_score(y_true_b37, y_pred_b37)
    prec_37 = precision_score(y_true_b37, y_pred_b37)
    f1_37 = f1_score(y_true_b37, y_pred_b37)
    kap_37 = cohen_kappa_score(y_true_b37, y_pred_b37)
    cm_37 = confusion_matrix(y_true_b37, y_pred_b37)

    print('\n--- 3. BINARY HEALTH ALERT EVALUATION (Threshold >= 37.5 ug/m3) ---')
    print(f'  Recall (Sensitivity): {rec_37:.4f} ({rec_37*100:.2f}%)')
    print(f'  Precision:            {prec_37:.4f} ({prec_37*100:.2f}%)')
    print(f'  F1 Score:             {f1_37:.4f}')
    print(f'  Cohen\'s Kappa (κ):     {kap_37:.4f}')
    print('  Confusion Matrix (Threshold 37.5):')
    print(f'    TN={cm_37[0,0]}, FP={cm_37[0,1]}')
    print(f'    FN={cm_37[1,0]}, TP={cm_37[1,1]}')

    # Binary Threshold Optimization for Maximum F1 and Recall
    print('\nThreshold Tuning Curve for Alert System:')
    threshold_results = []
    for th in [30.0, 32.5, 35.0, 37.5, 40.0]:
        y_pred_th = (preds >= th).astype(int)
        r = recall_score(y_true_b37, y_pred_th)
        p = precision_score(y_true_b37, y_pred_th)
        f = f1_score(y_true_b37, y_pred_th)
        k = cohen_kappa_score(y_true_b37, y_pred_th)
        threshold_results.append({'Decision Threshold': th, 'Recall': r, 'Precision': p, 'F1-Score': f, 'Kappa': k})
    th_df = pd.DataFrame(threshold_results)
    print(th_df.to_string(index=False))

    # 3. 5-Tier Thailand Official AQI Classification
    tier_names = ['1.ฟ้า (0-15)', '2.เขียว (15.1-25)', '3.เหลือง (25.1-37.5)', '4.ส้ม (37.6-75)', '5.แดง (>75)']
    bins = [-np.inf, 15.0, 25.0, 37.5, 75.0, np.inf]
    
    y_true_tier = pd.cut(y_test, bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)
    y_pred_tier = pd.cut(preds, bins=bins, labels=[0, 1, 2, 3, 4]).astype(int)

    f1_macro = f1_score(y_true_tier, y_pred_tier, average='macro')
    f1_weighted = f1_score(y_true_tier, y_pred_tier, average='weighted')
    rec_macro = recall_score(y_true_tier, y_pred_tier, average='macro')
    rec_weighted = recall_score(y_true_tier, y_pred_tier, average='weighted')
    kappa_std = cohen_kappa_score(y_true_tier, y_pred_tier)
    kappa_quad = cohen_kappa_score(y_true_tier, y_pred_tier, weights='quadratic')

    print('\n--- 4. 5-TIER THAILAND AQI CATEGORICAL EVALUATION ---')
    print(f'  Macro Recall:           {rec_macro:.4f} ({rec_macro*100:.2f}%)')
    print(f'  Weighted Recall:        {rec_weighted:.4f} ({rec_weighted*100:.2f}%)')
    print(f'  Macro F1-Score:         {f1_macro:.4f}')
    print(f'  Weighted F1-Score:      {f1_weighted:.4f}')
    print(f'  Cohen\'s Kappa (Standard):           {kappa_std:.4f}')
    print(f'  Cohen\'s Kappa (Quadratic Weighted): {kappa_quad:.4f}')

    print('\nDetailed Classification Report per Tier:')
    print(classification_report(y_true_tier, y_pred_tier, target_names=tier_names, digits=4))

    print('\n5x5 Confusion Matrix (Rows: Actual, Cols: Predicted):')
    cm_5x5 = confusion_matrix(y_true_tier, y_pred_tier)
    cm_df = pd.DataFrame(cm_5x5, index=[f'Actual {t}' for t in tier_names], columns=[f'Pred {t}' for t in tier_names])
    print(cm_df.to_string())

if __name__ == '__main__':
    run_fine_tuned_extra_trees()
