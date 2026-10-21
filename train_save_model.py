import pandas as pd
import numpy as np
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import ExtraTreesRegressor

def train_and_save():
    print('Loading pm25.csv...')
    df = pd.read_csv('pm25.csv')
    df['date'] = pd.to_datetime(df['date'])

    # Filter extreme sensor errors
    df = df[(df['pm25'] > 0) & (df['pm25'] <= 350)].copy()
    df = df.sort_values(['point_id', 'date']).reset_index(drop=True)

    # 1. Weather features
    rad = np.radians(df['wind_dir_deg'].fillna(180))
    df['wind_u'] = -df['wind_speed_station_m_s'] * np.sin(rad)
    df['wind_v'] = -df['wind_speed_station_m_s'] * np.cos(rad)
    df['ventilation_idx'] = df['wind_speed_station_m_s'] * df['mixing_height_station_m']
    df['stagnation_idx'] = 1.0 / (df['ventilation_idx'] + 1e-3)
    df['lst_missing'] = (df['lst_celsius'] == 0) | (df['lst_celsius'].isna()).astype(int)
    df['lst_imputed'] = df['lst_celsius'].replace(0, np.nan)
    df['lst_imputed'] = df.groupby('point_id')['lst_imputed'].ffill().bfill().fillna(26.0)

    for col in ['wind_speed_station_m_s', 'wind_u', 'wind_v', 'mixing_height_station_m', 
                'ventilation_idx', 'stagnation_idx', 'lst_imputed']:
        df[col] = df.groupby('point_id')[col].ffill().bfill().fillna(df[col].median())

    # 2. Hotspots
    for col in ['hotspot_10km_count', 'hotspot_25km_count', 'hotspot_50km_count', 'hotspot_province_count']:
        df[col] = df[col].fillna(0)
    df['hotspot_local_ratio'] = (df['hotspot_10km_count'] + 1) / (df['hotspot_50km_count'] + 1)

    # 3. Calendar
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month'] = df['date'].dt.month
    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    df['is_burning_season'] = df['month'].isin([1, 2, 3, 4]).astype(int)

    # 4. Spatio-temporal lags
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

    daily_prov = df.groupby('date')['pm25'].mean().rename('prov_pm25_mean')
    df = df.merge(daily_prov, on='date', how='left')
    df['prov_pm25_mean_lag_1'] = df.groupby('point_id')['prov_pm25_mean'].shift(1)

    df_model = df.dropna().reset_index(drop=True)

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

    X = df_model[feature_cols]
    y = df_model['pm25']

    print(f'Training ExtraTreesRegressor on {len(X)} samples...')
    model = ExtraTreesRegressor(
        n_estimators=250,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X, y)

    # Save model and feature list
    joblib.dump({'model': model, 'features': feature_cols}, 'pm25_model.joblib')
    print('Model saved to pm25_model.joblib')

    # Save station metadata
    # Extract district name from station name
    districts = ['เมือง', 'วังทอง', 'บางระกำ', 'พรหมพิราม', 'บางกระทุ่ม', 'นครไทย', 'ชาติตระการ', 'วัดโบสถ์', 'เนินมะปราง']
    
    stations_info = []
    for pid, g in df.groupby('point_id'):
        last_row = g.sort_values('date').iloc[-1]
        name = last_row['name']
        
        # detect district
        matched_dist = 'อำเภอเมืองพิษณุโลก'
        for d in districts:
            if f'อ.{d}' in name or f'อำเภอ{d}' in name or f'อ. {d}' in name:
                matched_dist = f'อ.{d}'
                break
            elif d in name:
                matched_dist = f'อ.{d}'

        stations_info.append({
            'point_id': str(pid),
            'name': name,
            'district': matched_dist,
            'lat': float(last_row['lat']),
            'lon': float(last_row['lon']),
            'last_pm25': float(round(last_row['pm25'], 2)),
            'avg_pm25': float(round(g['pm25'].mean(), 2)),
            'recent_lags': [float(round(v, 2)) for v in g.sort_values('date')['pm25'].tail(7).tolist()]
        })

    with open('stations.json', 'w', encoding='utf-8') as f:
        json.dump(stations_info, f, ensure_ascii=False, indent=2)

    print(f'Saved {len(stations_info)} stations to stations.json')

if __name__ == '__main__':
    train_and_save()
