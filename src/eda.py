"""
Exploratory Data Analysis module
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils import logger, save_figure

def perform_eda(df):
    """
    Comprehensive Exploratory Data Analysis
    """
    logger.info("="*50)
    logger.info("PART 2: EXPLORATORY DATA ANALYSIS")
    logger.info("="*50)
    
    # Basic statistics
    logger.info("\nBasic Statistical Summary:")
    logger.info(f"Dataset Shape: {df.shape}")
    logger.info(f"Date Range: {df['last_updated'].min()} to {df['last_updated'].max()}")
    logger.info(f"Number of Countries: {df['country'].nunique()}")
    logger.info(f"Number of Cities: {df['location_name'].nunique()}")
    
    # Numerical columns statistics
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    logger.info(f"\nNumerical Columns: {len(numerical_cols)}")
    
    # Create interactive dashboard
    create_interactive_dashboard(df)
    
    # Correlation analysis
    corr_matrix = correlation_analysis(df)
    
    # Time series patterns
    time_series_analysis(df)
    
    # Geographical analysis
    geographical_analysis(df)
    
    return corr_matrix

def create_interactive_dashboard(df):
    """Create interactive Plotly dashboard"""
    logger.info("Creating interactive dashboard...")
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=('Temperature Distribution', 'Temperature by Month', 'Precipitation by Month',
                       'Correlation Heatmap', 'Temperature by Country', 'Weather Patterns',
                       'Humidity vs Temperature', 'Wind Speed Analysis', 'Air Quality Trends'),
        specs=[[{'type': 'histogram'}, {'type': 'box'}, {'type': 'bar'}],
               [{'type': 'heatmap'}, {'type': 'scattergeo'}, {'type': 'scatter'}],
               [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}]]
    )
    
    # 1. Temperature Distribution
    fig.add_trace(
        go.Histogram(x=df['temperature_celsius'], nbinsx=50, name='Temperature',
                    marker_color='blue', opacity=0.7),
        row=1, col=1
    )
    
    # 2. Temperature by Month (Box plot)
    fig.add_trace(
        go.Box(x=df['month'], y=df['temperature_celsius'], name='Monthly Temp',
               boxmean='sd', marker_color='orange'),
        row=1, col=2
    )
    
    # 3. Precipitation by Month
    monthly_precip = df.groupby('month')['precip_mm'].mean().reset_index()
    fig.add_trace(
        go.Bar(x=monthly_precip['month'], y=monthly_precip['precip_mm'], 
               name='Avg Precipitation', marker_color='green'),
        row=1, col=3
    )
    
    # 4. Correlation Heatmap
    numeric_cols = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 
                   'precip_mm', 'visibility_km', 'uv_index']
    existing_cols = [col for col in numeric_cols if col in df.columns]
    corr_matrix = df[existing_cols].corr()
    
    fig.add_trace(
        go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, 
                   y=corr_matrix.columns, colorscale='RdBu',
                   zmin=-1, zmax=1, text=corr_matrix.values.round(2),
                   texttemplate='%{text}', textfont={"size": 10}),
        row=2, col=1
    )
    
    # 5. Geographical Temperature Distribution
    fig.add_trace(
        go.Scattergeo(
            lon=df['longitude'],
            lat=df['latitude'],
            text=df['temperature_celsius'].round(1).astype(str) + '°C',
            mode='markers',
            marker=dict(
                size=5,
                color=df['temperature_celsius'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Temperature °C")
            ),
            name='Global Temperatures'
        ),
        row=2, col=2
    )
    
    # 6. Temperature vs Humidity Scatter
    fig.add_trace(
        go.Scatter(
            x=df['humidity'],
            y=df['temperature_celsius'],
            mode='markers',
            marker=dict(
                size=3,
                color=df['precip_mm'] if 'precip_mm' in df.columns else 'blue',
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Precipitation mm")
            ),
            name='Temp vs Humidity'
        ),
        row=2, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=1200, 
        width=1500, 
        title_text="Comprehensive Weather Analysis Dashboard",
        showlegend=False
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="Month", row=1, col=2)
    fig.update_xaxes(title_text="Month", row=1, col=3)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Precipitation (mm)", row=1, col=3)
    
    fig.show()
    
    # Save dashboard
    try:
        fig.write_html("outputs/figures/interactive_dashboard.html")
        logger.info("Dashboard saved to outputs/figures/interactive_dashboard.html")
    except:
        logger.warning("Could not save dashboard (output directory may not exist)")

def correlation_analysis(df):
    """Perform correlation analysis"""
    logger.info("\nCorrelation Analysis:")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Find top correlations with temperature
    if 'temperature_celsius' in corr_matrix.columns:
        temp_correlations = corr_matrix['temperature_celsius'].sort_values(ascending=False)
        logger.info("Top correlations with temperature:")
        for col, corr in temp_correlations.head(6).items():
            if col != 'temperature_celsius':
                logger.info(f"  - {col}: {corr:.3f}")
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

def time_series_analysis(df):
    """Analyze time series patterns"""
    logger.info("\nTime Series Analysis:")
    
    # Aggregate by date
    daily_avg = df.groupby(df['last_updated'].dt.date)['temperature_celsius'].mean().reset_index()
    daily_avg['last_updated'] = pd.to_datetime(daily_avg['last_updated'])
    
    # Plot time series
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Daily average temperature
    axes[0, 0].plot(daily_avg['last_updated'], daily_avg['temperature_celsius'], 
                    color='blue', alpha=0.7, linewidth=1)
    axes[0, 0].set_title('Daily Average Temperature', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Monthly pattern
    monthly_avg = df.groupby('month')['temperature_celsius'].agg(['mean', 'std']).reset_index()
    axes[0, 1].errorbar(monthly_avg['month'], monthly_avg['mean'], 
                        yerr=monthly_avg['std'], capsize=5, marker='o')
    axes[0, 1].set_title('Monthly Temperature Pattern', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Temperature (°C)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Yearly trend
    yearly_avg = df.groupby('year')['temperature_celsius'].mean().reset_index()
    axes[1, 0].plot(yearly_avg['year'], yearly_avg['temperature_celsius'], 
                    marker='s', color='red', linewidth=2)
    axes[1, 0].set_title('Yearly Temperature Trend', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Year')
    axes[1, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Day of week pattern
    dow_avg = df.groupby('day_of_week')['temperature_celsius'].mean().reset_index()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 1].bar(days, dow_avg['temperature_celsius'], color='green', alpha=0.6)
    axes[1, 1].set_title('Temperature by Day of Week', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Day')
    axes[1, 1].set_ylabel('Temperature (°C)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Log insights
    logger.info(f"  - Hottest month: {monthly_avg.loc[monthly_avg['mean'].idxmax(), 'month']}")
    logger.info(f"  - Coldest month: {monthly_avg.loc[monthly_avg['mean'].idxmin(), 'month']}")
    logger.info(f"  - Temperature range: {df['temperature_celsius'].min():.1f}°C to {df['temperature_celsius'].max():.1f}°C")

def geographical_analysis(df):
    """Analyze geographical patterns"""
    logger.info("\nGeographical Analysis:")
    
    # Top 10 hottest countries
    country_temp = df.groupby('country')['temperature_celsius'].agg(['mean', 'count']).reset_index()
    country_temp = country_temp[country_temp['count'] > 100].sort_values('mean', ascending=False)
    
    logger.info("Top 10 hottest countries:")
    for idx, row in country_temp.head(10).iterrows():
        logger.info(f"  - {row['country']}: {row['mean']:.2f}°C ({row['count']} records)")
    
    # Top 10 coldest countries
    logger.info("\nTop 10 coldest countries:")
    for idx, row in country_temp.tail(10).iterrows():
        logger.info(f"  - {row['country']}: {row['mean']:.2f}°C ({row['count']} records)")
    
    # Create geographical plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))
    
    # Scatter plot on map background
    scatter = ax.scatter(df['longitude'], df['latitude'], 
                        c=df['temperature_celsius'], cmap='coolwarm',
                        s=10, alpha=0.6, vmin=-20, vmax=40)
    
    plt.colorbar(scatter, ax=ax, label='Temperature (°C)')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Global Temperature Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()