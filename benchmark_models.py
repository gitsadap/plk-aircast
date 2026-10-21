import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

def build_and_evaluate():
    print('--- LOADING DATA ---')
    df = pd.read_csv('pm25.csv')
    df['date'] = pd.to_datetime(df['date'])

    # 1. Outlier removal for target variable (Physical plausibility)
    # PM2.5 values > 350 ug/m3 or <= 0 are sensor malfunctions / freezing
    initial_len = len(df)
    df = df[(df['pm25'] > 0) & (df['pm25'] <= 350)].copy()
    print(f'Removed {initial_len - len(df)} sensor outlier rows. Remaining rows: {len(df)}')

    # Sort chronologically by station and date
    df = df.sort_values(['point_id', 'date']).reset_index(drop=True)

    # 2. Meteorological Feature Engineering
    # Circular wind direction to u and v vector components
    rad = np.radians(df['wind_dir_deg'].fillna(180))
    df['wind_u'] = -df['wind_speed_station_m_s'] * np.sin(rad)
    df['wind_v'] = -df['wind_speed_station_m_s'] * np.cos(rad)
    
    # Ventilation index (Boundary layer height * wind speed)
    df['ventilation_idx'] = df['wind_speed_station_m_s'] * df['mixing_height_station_m']
    
    # Atmospheric stagnation index (low wind + low mixing height)
    df['stagnation_idx'] = 1.0 / (df['ventilation_idx'] + 1e-3)

    # LST Handling (0 means cloud cover / missing satellite scan)
    df['lst_missing'] = (df['lst_celsius'] == 0) | (df['lst_celsius'].isna()).astype(int)
    # Impute missing LST with seasonal station mean or forward fill
    df['lst_imputed'] = df['lst_celsius'].replace(0, np.nan)
    df['lst_imputed'] = df.groupby('point_id')['lst_imputed'].ffill().bfill().fillna(26.0)

    # 3. Calendar & Seasonality Features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month'] = df['date'].dt.month
    df['is_burning_season'] = df['month'].isin([1, 2, 3, 4]).astype(int)

    # 4. Spatio-Temporal Lag Features (Grouped by Station)
    # In operational forecasting for Day t+1:
    # We use PM2.5 at t-1, t-2, t-3; Hotspots at t-1, t-2, t-3, rolling sums
    print('Generating spatio-temporal lag features per station...')
    
    # Fill weather missing values (only 19 rows)
    weather_cols = ['wind_speed_station_m_s', 'wind_u', 'wind_v', 'mixing_height_station_m', 
                    'ventilation_idx', 'stagnation_idx', 'lst_imputed']
    for col in weather_cols:
        df[col] = df.groupby('point_id')[col].ffill().bfill().fillna(df[col].median())

    hotspot_cols = ['hotspot_10km_count', 'hotspot_25km_count', 'hotspot_50km_count', 'hotspot_province_count']
    for col in hotspot_cols:
        df[col] = df[col].fillna(0)

    # Hotspot ratios and cumulative burning
    df['hotspot_local_ratio'] = (df['hotspot_10km_count'] + 1) / (df['hotspot_50km_count'] + 1)

    # Create lags per station
    # Ensure regular daily sequence or calculate calendar day lags
    lags = [1, 2, 3, 7]
    for lag in lags:
        df[f'pm25_lag_{lag}'] = df.groupby('point_id')['pm25'].shift(lag)
        for h_col in ['hotspot_25km_count', 'hotspot_province_count']:
            df[f'{h_col}_lag_{lag}'] = df.groupby('point_id')[h_col].shift(lag)

    # Rolling statistics for PM2.5 and Hotspots (using shift(1) to avoid data leakage from same day)
    df['pm25_roll_mean_3d'] = df.groupby('point_id')['pm25'].shift(1).rolling(3, min_periods=1).mean()
    df['pm25_roll_std_3d'] = df.groupby('point_id')['pm25'].shift(1).rolling(3, min_periods=1).std().fillna(0)
    df['pm25_roll_mean_7d'] = df.groupby('point_id')['pm25'].shift(1).rolling(7, min_periods=1).mean()
    df['hotspot_province_roll_sum_3d'] = df.groupby('point_id')['hotspot_province_count'].shift(1).rolling(3, min_periods=1).sum()
    df['hotspot_province_roll_sum_7d'] = df.groupby('point_id')['hotspot_province_count'].shift(1).rolling(7, min_periods=1).sum()

    # Drop initial rows where lags cannot be computed
    df_model = df.dropna().reset_index(drop=True)
    print(f'Dataset ready for modeling. Total samples: {len(df_model)}')

    # 5. Chronological Train-Test Split (Future Backtest)
    # Train: 2021-10 to 2025-05-31
    # Future Backtest Test: 2025-06-01 to 2026-10-08 (unseen future)
    split_date = '2025-06-01'
    train_mask = df_model['date'] < split_date
    test_mask = df_model['date'] >= split_date

    feature_cols = [
        'lat', 'lon',
        'sin_doy', 'cos_doy', 'is_burning_season',
        'wind_speed_station_m_s', 'wind_u', 'wind_v',
        'mixing_height_station_m', 'ventilation_idx', 'stagnation_idx',
        'lst_imputed', 'lst_missing',
        'hotspot_10km_count', 'hotspot_25km_count', 'hotspot_50km_count', 'hotspot_province_count',
        'hotspot_local_ratio',
        'hotspot_25km_count_lag_1', 'hotspot_25km_count_lag_2',
        'hotspot_province_count_lag_1', 'hotspot_province_count_lag_2', 'hotspot_province_count_lag_3',
        'hotspot_province_roll_sum_3d', 'hotspot_province_roll_sum_7d',
        'pm25_lag_1', 'pm25_lag_2', 'pm25_lag_3', 'pm25_lag_7',
        'pm25_roll_mean_3d', 'pm25_roll_std_3d', 'pm25_roll_mean_7d'
    ]

    X_train = df_model.loc[train_mask, feature_cols]
    y_train = df_model.loc[train_mask, 'pm25']
    X_test = df_model.loc[test_mask, feature_cols]
    y_test = df_model.loc[test_mask, 'pm25']

    t_min = df_model.loc[train_mask, 'date'].min().strftime('%Y-%m-%d')
    t_max = df_model.loc[train_mask, 'date'].max().strftime('%Y-%m-%d')
    v_min = df_model.loc[test_mask, 'date'].min().strftime('%Y-%m-%d')
    v_max = df_model.loc[test_mask, 'date'].max().strftime('%Y-%m-%d')
    print(f'Train period: {t_min} to {t_max}')
    print(f'Train size: {len(X_train)} samples')
    print(f'Test period:  {v_min} to {v_max}')
    print(f'Test size:  {len(X_test)} samples (Unseen Future Backtest)')

    # Models definition
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        'Extra Trees': ExtraTreesRegressor(n_estimators=150, max_depth=15, min_samples_split=5, random_state=42, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=250, learning_rate=0.04, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1),
        'XGBoost': XGBRegressor(n_estimators=250, learning_rate=0.04, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    }

    results = []
    test_preds = {}

    print('\n--- TRAINING & BACKTESTING MODELS ---')
    for name, model in models.items():
        print(f'Training {name}...')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds = np.clip(preds, 0, None) # PM2.5 cannot be negative
        test_preds[name] = preds

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results.append({
            'Model': name,
            'Future Test RMSE': rmse,
            'Future Test MAE': mae,
            'Future Test R2': r2
        })
        print(f'--> {name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.4f}')

    # Ensemble: Weighted Blending of LightGBM, XGBoost, and ExtraTrees
    ensemble_preds = (0.35 * test_preds['LightGBM'] + 
                      0.35 * test_preds['XGBoost'] + 
                      0.30 * test_preds['Extra Trees'])
    ens_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
    ens_mae = mean_absolute_error(y_test, ensemble_preds)
    ens_r2 = r2_score(y_test, ensemble_preds)

    results.append({
        'Model': 'Ensemble (Blend)',
        'Future Test RMSE': ens_rmse,
        'Future Test MAE': ens_mae,
        'Future Test R2': ens_r2
    })
    print(f'--> Ensemble (Blend): RMSE={ens_rmse:.3f}, MAE={ens_mae:.3f}, R2={ens_r2:.4f}')

    res_df = pd.DataFrame(results).sort_values('Future Test RMSE')
    print('\n=== FINAL BACKTEST BENCHMARK RESULTS ===')
    print(res_df.to_string(index=False))

    # Top feature importances from LightGBM
    lgb_model = models['LightGBM']
    importances = pd.Series(lgb_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print('\n=== TOP 15 MOST INFLUENTIAL FEATURES (LightGBM) ===')
    print(importances.head(15).to_string())

if __name__ == '__main__':
    build_and_evaluate()
