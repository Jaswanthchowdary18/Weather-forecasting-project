"""
Advanced Analyses Module for Weather Forecasting Project
Includes anomaly detection, climate analysis, feature importance, and spatial analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from src.utils import logger, save_figure, timer_decorator

@timer_decorator
def anomaly_detection_isolation_forest(df, contamination=0.1):
    """
    Detect anomalies using Isolation Forest algorithm
    """
    logger.info("\n🔍 Running Isolation Forest Anomaly Detection...")
    
    # Select features for anomaly detection
    feature_cols = ['temperature_celsius', 'humidity', 'pressure_mb', 
                    'wind_kph', 'precip_mm', 'visibility_km']
    existing_features = [f for f in feature_cols if f in df.columns]
    
    X = df[existing_features].dropna()
    
    logger.info(f"   • Using {len(existing_features)} features")
    logger.info(f"   • Data shape: {X.shape}")
    
    # Apply Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, 
                                 random_state=42, 
                                 n_jobs=-1)
    outliers = iso_forest.fit_predict(X)
    
    # Add results to dataframe
    df_anomaly = df.copy()
    df_anomaly['anomaly_if'] = outliers
    
    anomaly_count = (outliers == -1).sum()
    normal_count = (outliers == 1).sum()
    
    logger.info(f"\n   📊 Isolation Forest Results:")
    logger.info(f"      • Normal points: {normal_count} ({normal_count/len(outliers)*100:.2f}%)")
    logger.info(f"      • Anomalies: {anomaly_count} ({anomaly_count/len(outliers)*100:.2f}%)")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D visualization using first two features
    if len(existing_features) >= 2:
        normal = X[outliers == 1]
        anomaly = X[outliers == -1]
        
        axes[0].scatter(normal.iloc[:, 0], normal.iloc[:, 1], 
                       c='blue', label='Normal', alpha=0.5, s=5)
        axes[0].scatter(anomaly.iloc[:, 0], anomaly.iloc[:, 1], 
                       c='red', label='Anomaly', alpha=0.7, s=20)
        axes[0].set_xlabel(existing_features[0])
        axes[0].set_ylabel(existing_features[1])
        axes[0].set_title('Isolation Forest Anomalies')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Time series view if date available
    if 'last_updated' in df_anomaly.columns:
        df_sorted = df_anomaly.sort_values('last_updated')
        normal_ts = df_sorted[df_sorted['anomaly_if'] == 1]
        anomaly_ts = df_sorted[df_sorted['anomaly_if'] == -1]
        
        axes[1].scatter(normal_ts['last_updated'], normal_ts['temperature_celsius'],
                       c='blue', label='Normal', alpha=0.3, s=2)
        axes[1].scatter(anomaly_ts['last_updated'], anomaly_ts['temperature_celsius'],
                       c='red', label='Anomaly', alpha=0.7, s=10)
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Temperature (°C)')
        axes[1].set_title('Anomalies Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'isolation_forest_anomalies')
    plt.show()
    
    return df_anomaly, iso_forest

@timer_decorator
def anomaly_detection_dbscan(df, eps=0.5, min_samples=10):
    """
    Detect anomalies using DBSCAN clustering
    """
    logger.info("\n🔍 Running DBSCAN Anomaly Detection...")
    
    # Select and scale features
    feature_cols = ['temperature_celsius', 'humidity', 'pressure_mb', 
                    'wind_kph', 'precip_mm']
    existing_features = [f for f in feature_cols if f in df.columns]
    
    X = df[existing_features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Add results to dataframe
    df_anomaly = df.copy()
    df_anomaly['cluster'] = clusters
    df_anomaly['anomaly_db'] = (clusters == -1).astype(int)
    
    anomaly_count = (clusters == -1).sum()
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    
    logger.info(f"\n   📊 DBSCAN Results:")
    logger.info(f"      • Number of clusters: {n_clusters}")
    logger.info(f"      • Anomalies: {anomaly_count} ({anomaly_count/len(clusters)*100:.2f}%)")
    
    # Visualize using PCA if >2 dimensions
    if len(existing_features) > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=clusters, cmap='viridis', 
                             s=5, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title('DBSCAN Clustering Results (PCA Reduced)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        save_figure(plt.gcf(), 'dbscan_clusters')
        plt.show()
    
    return df_anomaly, dbscan

@timer_decorator
def climate_analysis_comprehensive(df):
    """
    Comprehensive climate analysis by region and season
    """
    logger.info("\n🌍 Running Comprehensive Climate Analysis...")
    
    # Define climate zones
    def get_climate_zone(lat):
        if lat > 66.5:
            return 'Arctic'
        elif lat > 23.5:
            return 'Temperate Northern'
        elif lat > -23.5:
            return 'Tropical'
        elif lat > -66.5:
            return 'Temperate Southern'
        else:
            return 'Antarctic'
    
    df['climate_zone'] = df['latitude'].apply(get_climate_zone)
    
    # Add season based on hemisphere
    df['month'] = pd.to_datetime(df['last_updated']).dt.month
    df['season'] = df.apply(lambda row: 
        'Winter' if row['month'] in [12, 1, 2] else
        'Spring' if row['month'] in [3, 4, 5] else
        'Summer' if row['month'] in [6, 7, 8] else 'Fall', axis=1)
    
    # Calculate comprehensive statistics by zone
    zone_stats = {}
    
    for zone in df['climate_zone'].unique():
        zone_data = df[df['climate_zone'] == zone]
        
        stats = {
            'temp_mean': zone_data['temperature_celsius'].mean(),
            'temp_std': zone_data['temperature_celsius'].std(),
            'temp_min': zone_data['temperature_celsius'].min(),
            'temp_max': zone_data['temperature_celsius'].max(),
            'humidity_mean': zone_data['humidity'].mean() if 'humidity' in zone_data else None,
            'pressure_mean': zone_data['pressure_mb'].mean() if 'pressure_mb' in zone_data else None,
            'precip_mean': zone_data['precip_mm'].mean() if 'precip_mm' in zone_data else None,
            'wind_mean': zone_data['wind_kph'].mean() if 'wind_kph' in zone_data else None,
            'count': len(zone_data)
        }
        zone_stats[zone] = stats
        
        logger.info(f"\n   📊 {zone} Zone:")
        logger.info(f"      • Temperature: {stats['temp_mean']:.1f}°C ± {stats['temp_std']:.1f}°C")
        logger.info(f"      • Range: {stats['temp_min']:.1f}°C to {stats['temp_max']:.1f}°C")
        logger.info(f"      • Records: {stats['count']:,}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Temperature by zone
    zones = list(zone_stats.keys())
    temp_means = [zone_stats[z]['temp_mean'] for z in zones]
    
    axes[0,0].bar(zones, temp_means, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B8F5E'])
    axes[0,0].set_title('Average Temperature by Climate Zone')
    axes[0,0].set_xlabel('Climate Zone')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Temperature distribution by zone
    for i, zone in enumerate(zones):
        zone_data = df[df['climate_zone'] == zone]['temperature_celsius']
        axes[0,1].hist(zone_data, bins=30, alpha=0.5, label=zone)
    axes[0,1].set_title('Temperature Distribution by Zone')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Seasonal patterns by zone
    seasonal = df.groupby(['climate_zone', 'season'])['temperature_celsius'].mean().unstack()
    seasonal.plot(kind='bar', ax=axes[1,0])
    axes[1,0].set_title('Seasonal Temperature Patterns')
    axes[1,0].set_xlabel('Climate Zone')
    axes[1,0].set_ylabel('Temperature (°C)')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Humidity vs Temperature by zone
    if 'humidity' in df.columns:
        for zone in zones:
            zone_data = df[df['climate_zone'] == zone]
            axes[1,1].scatter(zone_data['humidity'], zone_data['temperature_celsius'], 
                             alpha=0.3, s=1, label=zone)
        axes[1,1].set_title('Humidity vs Temperature by Zone')
        axes[1,1].set_xlabel('Humidity (%)')
        axes[1,1].set_ylabel('Temperature (°C)')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'climate_analysis_comprehensive')
    plt.show()
    
    return zone_stats

@timer_decorator
def environmental_impact_detailed(df):
    """
    Detailed environmental impact analysis including air quality
    """
    logger.info("\n🌫️ Running Detailed Environmental Impact Analysis...")
    
    # Find air quality columns
    air_cols = [col for col in df.columns if 'air_quality' in col or 'PM' in col]
    
    if not air_cols:
        logger.warning("   No air quality data found")
        return None
    
    air_col = air_cols[0]
    logger.info(f"   • Analyzing: {air_col}")
    
    # Calculate correlations
    env_factors = ['temperature_celsius', 'humidity', 'pressure_mb', 
                   'wind_kph', 'precip_mm']
    existing_factors = [f for f in env_factors if f in df.columns]
    
    correlations = {}
    for factor in existing_factors:
        corr = df[factor].corr(df[air_col])
        correlations[factor] = corr
        logger.info(f"      • {factor}: {corr:.3f}")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Air quality distribution
    axes[0,0].hist(df[air_col].dropna(), bins=50, edgecolor='black', 
                   alpha=0.7, color='#2E86AB')
    axes[0,0].set_title(f'{air_col} Distribution')
    axes[0,0].set_xlabel(air_col)
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].grid(True, alpha=0.3)
    
    # Temperature vs Air Quality
    axes[0,1].scatter(df['temperature_celsius'], df[air_col], 
                      alpha=0.3, s=1, color='#A23B72')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel(air_col)
    axes[0,1].set_title('Temperature vs Air Quality')
    axes[0,1].grid(True, alpha=0.3)
    
    # Correlation heatmap
    if len(existing_factors) > 1:
        corr_matrix = df[existing_factors + [air_col]].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                   fmt='.2f', ax=axes[1,0])
        axes[1,0].set_title('Correlation Matrix')
    
    # Time series of air quality
    if 'last_updated' in df.columns:
        df_sorted = df.sort_values('last_updated')
        axes[1,1].plot(df_sorted['last_updated'], df_sorted[air_col], 
                       alpha=0.5, linewidth=0.5, color='#F18F01')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel(air_col)
        axes[1,1].set_title('Air Quality Over Time')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'environmental_impact')
    plt.show()
    
    return correlations

@timer_decorator
def feature_importance_multiple_methods(df, target='temperature_celsius'):
    """
    Calculate feature importance using multiple methods
    """
    logger.info("\n⭐ Running Feature Importance Analysis...")
    
    # Prepare features
    feature_cols = ['humidity', 'pressure_mb', 'wind_kph', 'precip_mm', 
                    'visibility_km', 'uv_index', 'month', 'day_of_week']
    
    if 'air_quality_PM2.5' in df.columns:
        feature_cols.append('air_quality_PM2.5')
    
    existing_features = [f for f in feature_cols if f in df.columns]
    
    # Prepare data
    X = df[existing_features].dropna()
    y = df.loc[X.index, target]
    
    logger.info(f"   • Analyzing {len(existing_features)} features")
    
    # Method 1: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importance = dict(zip(existing_features, rf.feature_importances_))
    
    # Method 2: Correlation
    correlations = {}
    for feature in existing_features:
        corr = abs(df[feature].corr(df[target]))
        correlations[feature] = corr
    
    # Method 3: Mutual Information (simplified - using correlation as proxy)
    mi_importance = correlations.copy()
    
    # Combine methods
    importance_df = pd.DataFrame({
        'Random Forest': rf_importance,
        'Correlation': correlations,
        'MI (proxy)': mi_importance
    })
    
    importance_df['Average'] = importance_df.mean(axis=1)
    importance_df = importance_df.sort_values('Average', ascending=False)
    
    logger.info("\n   📊 Top 5 Features (by average importance):")
    for i, (feature, row) in enumerate(importance_df.head(5).iterrows(), 1):
        logger.info(f"      {i}. {feature}: {row['Average']:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Random Forest
    importance_df['Random Forest'].plot(kind='barh', ax=axes[0,0], 
                                        color='#2E86AB')
    axes[0,0].set_title('Random Forest Importance')
    axes[0,0].set_xlabel('Importance')
    axes[0,0].invert_yaxis()
    axes[0,0].grid(True, alpha=0.3)
    
    # Correlation
    importance_df['Correlation'].plot(kind='barh', ax=axes[0,1], 
                                      color='#A23B72')
    axes[0,1].set_title('Absolute Correlation')
    axes[0,1].set_xlabel('|Correlation|')
    axes[0,1].invert_yaxis()
    axes[0,1].grid(True, alpha=0.3)
    
    # Average importance
    importance_df['Average'].plot(kind='barh', ax=axes[1,0], 
                                  color='#F18F01')
    axes[1,0].set_title('Average Importance')
    axes[1,0].set_xlabel('Average Score')
    axes[1,0].invert_yaxis()
    axes[1,0].grid(True, alpha=0.3)
    
    # Comparison plot
    importance_df[['Random Forest', 'Correlation', 'MI (proxy)']].plot(
        kind='bar', ax=axes[1,1])
    axes[1,1].set_title('Feature Importance Comparison')
    axes[1,1].set_xlabel('Feature')
    axes[1,1].set_ylabel('Importance Score')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'feature_importance_comprehensive')
    plt.show()
    
    return importance_df['Average'].to_dict()

@timer_decorator
def spatial_analysis_advanced(df):
    """
    Advanced spatial analysis with clustering and patterns
    """
    logger.info("\n🗺️ Running Advanced Spatial Analysis...")
    
    # Create spatial clusters based on weather patterns
    spatial_features = ['temperature_celsius', 'humidity', 'pressure_mb', 
                       'wind_kph', 'latitude', 'longitude']
    existing_features = [f for f in spatial_features if f in df.columns]
    
    if len(existing_features) >= 3:
        X = df[existing_features].dropna()
        X_scaled = StandardScaler().fit_transform(X)
        
        # K-means clustering (simplified)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        df['spatial_cluster'] = kmeans.fit_predict(X_scaled)
        
        logger.info(f"\n   📊 Spatial Clustering Results:")
        for i in range(5):
            cluster_size = (df['spatial_cluster'] == i).sum()
            logger.info(f"      • Cluster {i}: {cluster_size} points")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Geographic clusters
        scatter = axes[0].scatter(df['longitude'], df['latitude'], 
                                 c=df['spatial_cluster'], cmap='viridis',
                                 s=1, alpha=0.5)
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].set_title('Geographic Weather Clusters')
        plt.colorbar(scatter, ax=axes[0])
        
        # Temperature by cluster
        cluster_temp = df.groupby('spatial_cluster')['temperature_celsius'].mean()
        axes[1].bar(range(len(cluster_temp)), cluster_temp.values, 
                   color=plt.cm.viridis(np.linspace(0, 1, 5)))
        axes[1].set_xlabel('Cluster')
        axes[1].set_ylabel('Average Temperature (°C)')
        axes[1].set_title('Temperature by Spatial Cluster')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(plt.gcf(), 'spatial_clusters')
        plt.show()
        
        return cluster_temp.to_dict()
    
    return None

@timer_decorator
def extreme_weather_analysis(df):
    """
    Analyze extreme weather events
    """
    logger.info("\n⚠️ Running Extreme Weather Analysis...")
    
    # Define extreme thresholds
    temp_mean = df['temperature_celsius'].mean()
    temp_std = df['temperature_celsius'].std()
    
    extreme_hot = df[df['temperature_celsius'] > temp_mean + 2*temp_std]
    extreme_cold = df[df['temperature_celsius'] < temp_mean - 2*temp_std]
    
    logger.info(f"\n   📊 Extreme Weather Events:")
    logger.info(f"      • Extreme Hot (> {temp_mean + 2*temp_std:.1f}°C): {len(extreme_hot)} records")
    logger.info(f"      • Extreme Cold (< {temp_mean - 2*temp_std:.1f}°C): {len(extreme_cold)} records")
    
    # Find top extreme locations
    if len(extreme_hot) > 0:
        hottest = extreme_hot.nlargest(5, 'temperature_celsius')
        logger.info("\n      🔥 Hottest Locations:")
        for _, row in hottest.iterrows():
            logger.info(f"         • {row['country']}: {row['temperature_celsius']:.1f}°C")
    
    if len(extreme_cold) > 0:
        coldest = extreme_cold.nsmallest(5, 'temperature_celsius')
        logger.info("\n      ❄️ Coldest Locations:")
        for _, row in coldest.iterrows():
            logger.info(f"         • {row['country']}: {row['temperature_celsius']:.1f}°C")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Extreme events over time
    if 'last_updated' in df.columns:
        df_sorted = df.sort_values('last_updated')
        axes[0].plot(df_sorted['last_updated'], df_sorted['temperature_celsius'],
                    alpha=0.3, linewidth=0.5, color='gray')
        axes[0].scatter(extreme_hot['last_updated'], extreme_hot['temperature_celsius'],
                       color='red', s=5, label='Extreme Hot')
        axes[0].scatter(extreme_cold['last_updated'], extreme_cold['temperature_celsius'],
                       color='blue', s=5, label='Extreme Cold')
        axes[0].axhline(y=temp_mean + 2*temp_std, color='red', linestyle='--', alpha=0.5)
        axes[0].axhline(y=temp_mean - 2*temp_std, color='blue', linestyle='--', alpha=0.5)
        axes[0].set_title('Extreme Weather Events Over Time')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Temperature (°C)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Geographic distribution
    axes[1].scatter(df['longitude'], df['latitude'], 
                   c='gray', s=1, alpha=0.3, label='Normal')
    if len(extreme_hot) > 0:
        axes[1].scatter(extreme_hot['longitude'], extreme_hot['latitude'],
                       c='red', s=5, label='Extreme Hot')
    if len(extreme_cold) > 0:
        axes[1].scatter(extreme_cold['longitude'], extreme_cold['latitude'],
                       c='blue', s=5, label='Extreme Cold')
    axes[1].set_xlabel('Longitude')
    axes[1].set_ylabel('Latitude')
    axes[1].set_title('Geographic Distribution of Extreme Events')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'extreme_weather')
    plt.show()
    
    return {
        'extreme_hot_count': len(extreme_hot),
        'extreme_cold_count': len(extreme_cold),
        'hot_threshold': temp_mean + 2*temp_std,
        'cold_threshold': temp_mean - 2*temp_std
    }

@timer_decorator
def trend_decomposition(data):
    """
    Decompose time series into trend, seasonal, and residual components
    """
    logger.info("\n📉 Running Trend Decomposition...")
    
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    try:
        # Decompose time series
        decomposition = seasonal_decompose(data, model='additive', period=365)
        
        # Calculate trend strength
        trend_strength = max(0, 1 - np.var(decomposition.resid) / np.var(decomposition.trend + decomposition.resid))
        seasonal_strength = max(0, 1 - np.var(decomposition.resid) / np.var(decomposition.seasonal + decomposition.resid))
        
        logger.info(f"\n   📊 Decomposition Results:")
        logger.info(f"      • Trend Strength: {trend_strength:.3f}")
        logger.info(f"      • Seasonal Strength: {seasonal_strength:.3f}")
        logger.info(f"      • Residual Variance: {np.var(decomposition.resid):.4f}")
        
        # Visualize
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        decomposition.observed.plot(ax=axes[0], title='Original Time Series', color='#2E86AB')
        decomposition.trend.plot(ax=axes[1], title='Trend Component', color='#A23B72')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal Component', color='#F18F01')
        decomposition.resid.plot(ax=axes[3], title='Residual Component', color='#C73E1D')
        
        for ax in axes:
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_figure(plt.gcf(), 'trend_decomposition')
        plt.show()
        
        return {
            'trend_strength': trend_strength,
            'seasonal_strength': seasonal_strength,
            'residual_variance': np.var(decomposition.resid)
        }
        
    except Exception as e:
        logger.error(f"   Decomposition failed: {e}")
        return None