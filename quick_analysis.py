# quick_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("="*60)
print("QUICK WEATHER ANALYSIS")
print("="*60)

# Load data
print("\n📊 Loading data...")
df = pd.read_csv('data/Global Weather Repository.csv')
print(f"✅ Loaded {len(df):,} rows")
print(f"✅ Found {len(df.columns)} columns")

# Basic statistics
print("\n📈 Basic Statistics:")
print(f"   Average temperature: {df['temperature_celsius'].mean():.2f}°C")
print(f"   Min temperature: {df['temperature_celsius'].min():.2f}°C")
print(f"   Max temperature: {df['temperature_celsius'].max():.2f}°C")
print(f"   Number of countries: {df['country'].nunique()}")

# Create figures
print("\n🎨 Creating visualizations...")

# 1. Temperature Distribution
plt.figure(figsize=(12, 6))
plt.hist(df['temperature_celsius'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.title('Global Temperature Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/temperature_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ temperature_distribution.png")

# 2. Top 20 Hottest Countries
top_countries = df.groupby('country')['temperature_celsius'].mean().sort_values(ascending=False).head(20)
plt.figure(figsize=(14, 7))
top_countries.plot(kind='bar', color='coral')
plt.title('Top 20 Hottest Countries (Average Temperature)', fontsize=16, fontweight='bold')
plt.xlabel('Country')
plt.ylabel('Average Temperature (°C)')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/figures/top_20_hottest_countries.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ top_20_hottest_countries.png")

# 3. Temperature by Latitude
plt.figure(figsize=(12, 6))
plt.scatter(df['latitude'], df['temperature_celsius'], alpha=0.1, s=1, c='green')
plt.title('Temperature vs Latitude', fontsize=16, fontweight='bold')
plt.xlabel('Latitude')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.savefig('outputs/figures/temperature_vs_latitude.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ temperature_vs_latitude.png")

# 4. Temperature by Month
df['last_updated'] = pd.to_datetime(df['last_updated'])
df['month'] = df['last_updated'].dt.month
monthly_temp = df.groupby('month')['temperature_celsius'].agg(['mean', 'std'])

plt.figure(figsize=(12, 6))
plt.errorbar(monthly_temp.index, monthly_temp['mean'], yerr=monthly_temp['std'], 
             marker='o', capsize=5, linewidth=2, color='purple')
plt.title('Monthly Temperature Pattern (with variability)', fontsize=16, fontweight='bold')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 13))
plt.savefig('outputs/figures/monthly_temperature.png', dpi=300, bbox_inches='tight')
plt.close()
print("   ✅ monthly_temperature.png")

# 5. Correlation Heatmap
numeric_cols = ['temperature_celsius', 'humidity', 'pressure_mb', 'wind_kph', 
                'precip_mm', 'visibility_km', 'uv_index']
existing_cols = [col for col in numeric_cols if col in df.columns]

if len(existing_cols) > 1:
    plt.figure(figsize=(10, 8))
    corr_matrix = df[existing_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', square=True, linewidths=1)
    plt.title('Weather Parameters Correlation', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ correlation_heatmap.png")

# 6. Humidity vs Temperature
if 'humidity' in df.columns:
    plt.figure(figsize=(12, 6))
    plt.scatter(df['humidity'], df['temperature_celsius'], alpha=0.1, s=1, c='red')
    plt.title('Humidity vs Temperature', fontsize=16, fontweight='bold')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Temperature (°C)')
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/figures/humidity_vs_temperature.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ humidity_vs_temperature.png")

print("\n" + "="*60)
print("✅ ANALYSIS COMPLETE!")
print("="*60)
print(f"\n📁 All visualizations saved to: {os.path.abspath('outputs/figures')}")

# List all created files
print("\n📄 Files created:")
for f in os.listdir('outputs/figures'):
    print(f"   - {f}")

print("\n" + "="*60)