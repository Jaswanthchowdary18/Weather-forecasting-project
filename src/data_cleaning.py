"""
Data cleaning and preprocessing module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils import logger, reduce_memory_usage

def clean_and_preprocess_data(df):
    """
    Comprehensive data cleaning and preprocessing function
    """
    logger.info("="*50)
    logger.info("PART 1: DATA CLEANING & PREPROCESSING")
    logger.info("="*50)
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Reduce memory usage
    df_clean = reduce_memory_usage(df_clean)
    
    # 1. Handle datetime features
    logger.info("Processing datetime features...")
    df_clean['last_updated'] = pd.to_datetime(df_clean['last_updated'])
    df_clean['year'] = df_clean['last_updated'].dt.year
    df_clean['month'] = df_clean['last_updated'].dt.month
    df_clean['day'] = df_clean['last_updated'].dt.day
    df_clean['day_of_week'] = df_clean['last_updated'].dt.dayofweek
    df_clean['quarter'] = df_clean['last_updated'].dt.quarter
    df_clean['week_of_year'] = df_clean['last_updated'].dt.isocalendar().week
    
    # 2. Handle missing values
    logger.info("\nHandling missing values...")
    
    # For numerical columns: fill with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            logger.info(f"  - {col}: filled {df[col].isnull().sum()} missing values with median ({median_val:.2f})")
    
    # For categorical columns: fill with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            logger.info(f"  - {col}: filled {df[col].isnull().sum()} missing values with mode ({mode_val})")
    
    # 3. Handle outliers using IQR method
    logger.info("\nDetecting and handling outliers...")
    outlier_cols = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 
                   'precip_mm', 'visibility_km', 'uv_index']
    
    outlier_stats = {}
    for col in outlier_cols:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
            if len(outliers) > 0:
                outlier_stats[col] = len(outliers)
                logger.info(f"  - {col}: {len(outliers)} outliers detected ({len(outliers)/len(df_clean)*100:.2f}%)")
                # Cap outliers
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
    
    # 4. Normalize numerical features
    logger.info("\nNormalizing numerical features...")
    scaler = StandardScaler()
    features_to_normalize = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 
                            'precip_mm', 'visibility_km', 'uv_index']
    
    if 'air_quality_PM2.5' in df_clean.columns:
        features_to_normalize.append('air_quality_PM2.5')
    
    existing_features = [f for f in features_to_normalize if f in df_clean.columns]
    df_clean[existing_features] = scaler.fit_transform(df_clean[existing_features])
    logger.info(f"  - Normalized {len(existing_features)} features using StandardScaler")
    
    # 5. Create additional derived features
    logger.info("\nCreating derived features...")
    
    # Temperature categories
    df_clean['temp_category'] = pd.cut(df_clean['temperature_celsius'] if 'temperature_celsius' in df_clean.columns else df['temperature_celsius'], 
                                       bins=[-float('inf'), 0, 10, 20, 30, float('inf')],
                                       labels=['Freezing', 'Cold', 'Mild', 'Warm', 'Hot'])
    logger.info("  - Created 'temp_category'")
    
    # Weather severity index
    if all(col in df_clean.columns for col in ['wind_kph', 'precip_mm']):
        df_clean['weather_severity'] = (df_clean['wind_kph'] * 0.4 + df_clean['precip_mm'] * 0.6)
        logger.info("  - Created 'weather_severity' index")
    
    # Humidity categories
    if 'humidity' in df_clean.columns:
        df_clean['humidity_category'] = pd.cut(df_clean['humidity'], 
                                              bins=[0, 30, 60, 100],
                                              labels=['Dry', 'Comfortable', 'Humid'])
        logger.info("  - Created 'humidity_category'")
    
    # Temperature range (if both min and max exist)
    if all(col in df_clean.columns for col in ['temperature_celsius', 'temperature_fahrenheit']):
        df_clean['temp_range'] = abs(df_clean['temperature_celsius'] - df_clean['temperature_fahrenheit'])
        logger.info("  - Created 'temp_range'")
    
    logger.info(f"\nFinal dataset shape: {df_clean.shape}")
    logger.info(f"Final dataset memory usage: {df_clean.memory_usage().sum() / 1024**2:.2f} MB")
    
    return df_clean, scaler, outlier_stats