"""
Comprehensive Forecasting Models Module
Includes ARIMA, SARIMA, LSTM, Prophet, and Ensemble methods
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Facebook Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except:
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not installed. Install with: pip install prophet")

from src.utils import logger, save_figure, timer_decorator

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

@timer_decorator
def prepare_time_series_data(data, target_col='temperature_celsius'):
    """
    Prepare and analyze time series data
    """
    logger.info("\n📊 Preparing Time Series Data...")
    
    # Ensure datetime index
    if 'last_updated' in data.columns:
        ts_data = data.set_index('last_updated').sort_index()
    else:
        ts_data = data.copy()
    
    # Resample to daily frequency
    daily_data = ts_data[target_col].resample('D').mean()
    
    # Fill missing values
    daily_data = daily_data.interpolate(method='linear')
    
    logger.info(f"   • Time series length: {len(daily_data)} days")
    logger.info(f"   • Date range: {daily_data.index.min()} to {daily_data.index.max()}")
    logger.info(f"   • Frequency: {pd.infer_freq(daily_data.index)}")
    
    # Check stationarity
    result = adfuller(daily_data.dropna())
    logger.info(f"\n   • ADF Statistic: {result[0]:.4f}")
    logger.info(f"   • p-value: {result[1]:.4f}")
    
    if result[1] < 0.05:
        logger.info("   • Series is stationary ✓")
    else:
        logger.info("   • Series is non-stationary (will apply differencing)")
    
    # Decompose time series
    try:
        decomposition = seasonal_decompose(daily_data, model='additive', period=365)
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))
        daily_data.plot(ax=axes[0], title='Original Time Series', color='#2E86AB')
        decomposition.trend.plot(ax=axes[1], title='Trend', color='#A23B72')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonality', color='#F18F01')
        decomposition.resid.plot(ax=axes[3], title='Residuals', color='#C73E1D')
        
        plt.tight_layout()
        save_figure(plt.gcf(), 'time_series_decomposition')
        plt.show()
        
    except Exception as e:
        logger.warning(f"Could not decompose time series: {e}")
        decomposition = None
    
    return daily_data, decomposition

@timer_decorator
def build_arima_model(data, forecast_days=30):
    """
    Build ARIMA model for forecasting
    """
    logger.info("\n📈 Building ARIMA Model...")
    
    # Split data
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    logger.info(f"   • Training data: {len(train)} days")
    logger.info(f"   • Testing data: {len(test)} days")
    
    # Try different ARIMA orders
    best_aic = np.inf
    best_order = None
    best_model = None
    
    orders = [(1,0,0), (2,0,0), (3,0,0), (1,1,0), (2,1,0), (3,1,0)]
    
    for order in orders:
        try:
            model = ARIMA(train, order=order)
            fitted = model.fit()
            if fitted.aic < best_aic:
                best_aic = fitted.aic
                best_order = order
                best_model = fitted
        except:
            continue
    
    logger.info(f"   • Best ARIMA order: {best_order}")
    logger.info(f"   • AIC: {best_aic:.2f}")
    
    # Make predictions
    predictions = best_model.forecast(steps=len(test))
    
    # Calculate metrics
    mse = mean_squared_error(test, predictions)
    mae = mean_absolute_error(test, predictions)
    r2 = r2_score(test, predictions)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    logger.info(f"\n   📊 ARIMA Performance:")
    logger.info(f"      • MSE: {mse:.4f}")
    logger.info(f"      • MAE: {mae:.4f}")
    logger.info(f"      • RMSE: {rmse:.4f}")
    logger.info(f"      • R²: {r2:.4f}")
    
    # Forecast future
    future_forecast = best_model.forecast(steps=forecast_days)
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Training/Test plot
    axes[0].plot(train.index[-100:], train[-100:], label='Training', color='#2E86AB')
    axes[0].plot(test.index, test, label='Actual', color='#3B8F5E')
    axes[0].plot(test.index, predictions, label='ARIMA Predictions', 
                 color='#C73E1D', linestyle='--')
    axes[0].set_title('ARIMA Model: Training and Test Performance')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Temperature (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Future forecast
    future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), 
                                 periods=forecast_days)
    axes[1].plot(data.index[-100:], data[-100:], label='Historical', color='#2E86AB')
    axes[1].plot(future_dates, future_forecast, label=f'{forecast_days}-Day Forecast', 
                 color='#F18F01', linewidth=2)
    axes[1].fill_between(future_dates, 
                         future_forecast - 2*np.std(predictions),
                         future_forecast + 2*np.std(predictions),
                         color='#F18F01', alpha=0.2, label='95% Confidence Interval')
    axes[1].set_title(f'ARIMA Model: {forecast_days}-Day Forecast')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'arima_forecast')
    plt.show()
    
    return best_model, predictions, metrics

@timer_decorator
def build_sarima_model(data, forecast_days=30):
    """
    Build SARIMA model with seasonality
    """
    logger.info("\n📊 Building SARIMA Model...")
    
    # Split data
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # SARIMA order (p,d,q)(P,D,Q,s)
    # Assuming weekly seasonality (s=7)
    try:
        model = SARIMAX(train, 
                       order=(1, 1, 1), 
                       seasonal_order=(1, 1, 1, 7),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted_model = model.fit(disp=False)
        
        # Make predictions
        predictions = fitted_model.forecast(steps=len(test))
        
        # Calculate metrics
        mse = mean_squared_error(test, predictions)
        mae = mean_absolute_error(test, predictions)
        r2 = r2_score(test, predictions)
        rmse = np.sqrt(mse)
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"\n   📊 SARIMA Performance:")
        logger.info(f"      • MSE: {mse:.4f}")
        logger.info(f"      • MAE: {mae:.4f}")
        logger.info(f"      • RMSE: {rmse:.4f}")
        logger.info(f"      • R²: {r2:.4f}")
        
        # Forecast future
        future_forecast = fitted_model.forecast(steps=forecast_days)
        
        return fitted_model, predictions, metrics
        
    except Exception as e:
        logger.error(f"SARIMA model failed: {e}")
        return None, None, None

@timer_decorator
def build_lstm_model(data, forecast_days=30, seq_length=10):
    """
    Build advanced LSTM model with bidirectional layers
    """
    logger.info("\n🧠 Building Bidirectional LSTM Model...")
    
    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    # Create sequences
    X, y = create_sequences(data_scaled.flatten(), seq_length)
    
    logger.info(f"   • Input shape: {X.shape}")
    logger.info(f"   • Output shape: {y.shape}")
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Reshape for LSTM [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build advanced LSTM model
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu'), 
                     input_shape=(seq_length, 1)),
        Dropout(0.3),
        
        Bidirectional(LSTM(64, return_sequences=True, activation='relu')),
        Dropout(0.3),
        
        Bidirectional(LSTM(32, activation='relu')),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    # Compile with advanced optimizer
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=15, 
                               restore_best_weights=True, verbose=0)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=5, min_lr=0.0001, verbose=0)
    
    # Train model
    logger.info("   • Training LSTM (this may take 3-5 minutes)...")
    history = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size=32,
                       validation_split=0.1,
                       callbacks=[early_stop, reduce_lr],
                       verbose=0)
    
    logger.info(f"   • Training completed after {len(history.history['loss'])} epochs")
    
    # Make predictions
    train_pred = model.predict(X_train, verbose=0)
    test_pred = model.predict(X_test, verbose=0)
    
    # Inverse transform
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate metrics
    mse = mean_squared_error(y_test_actual, test_pred)
    mae = mean_absolute_error(y_test_actual, test_pred)
    r2 = r2_score(y_test_actual, test_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    logger.info(f"\n   📊 LSTM Performance:")
    logger.info(f"      • MSE: {mse:.4f}")
    logger.info(f"      • MAE: {mae:.4f}")
    logger.info(f"      • RMSE: {rmse:.4f}")
    logger.info(f"      • R²: {r2:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training history
    axes[0,0].plot(history.history['loss'], label='Training Loss')
    axes[0,0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0,0].set_title('LSTM Training History')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Predictions vs Actual
    axes[0,1].plot(y_test_actual[:100], label='Actual', color='#2E86AB')
    axes[0,1].plot(test_pred[:100], label='LSTM Predictions', color='#C73E1D')
    axes[0,1].set_title('LSTM: Actual vs Predicted (First 100 samples)')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('Temperature (°C)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Error distribution
    errors = y_test_actual.flatten() - test_pred.flatten()
    axes[1,0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='#A23B72')
    axes[1,0].set_title('Prediction Error Distribution')
    axes[1,0].set_xlabel('Error (°C)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1,1].scatter(y_test_actual, test_pred, alpha=0.5, s=5, color='#3B8F5E')
    axes[1,1].plot([y_test_actual.min(), y_test_actual.max()],
                   [y_test_actual.min(), y_test_actual.max()],
                   'r--', lw=2)
    axes[1,1].set_title('Actual vs Predicted Scatter')
    axes[1,1].set_xlabel('Actual Temperature (°C)')
    axes[1,1].set_ylabel('Predicted Temperature (°C)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'lstm_results')
    plt.show()
    
    return model, test_pred, metrics, history

@timer_decorator
def build_prophet_model(data, forecast_days=30):
    """
    Build Facebook Prophet model
    """
    if not PROPHET_AVAILABLE:
        logger.warning("Prophet not available. Skipping...")
        return None, None, None
    
    logger.info("\n🔮 Building Prophet Model...")
    
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': data.index,
        'y': data.values
    })
    
    # Split data
    train_size = int(len(prophet_df) * 0.8)
    train = prophet_df[:train_size]
    test = prophet_df[train_size:]
    
    # Create and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',
        changepoint_prior_scale=0.05
    )
    
    model.fit(train)
    
    # Make predictions
    future = model.make_future_dataframe(periods=len(test) + forecast_days)
    forecast = model.predict(future)
    
    # Extract test predictions
    predictions = forecast['yhat'].iloc[train_size:train_size+len(test)].values
    
    # Calculate metrics
    mse = mean_squared_error(test['y'].values, predictions)
    mae = mean_absolute_error(test['y'].values, predictions)
    r2 = r2_score(test['y'].values, predictions)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    logger.info(f"\n   📊 Prophet Performance:")
    logger.info(f"      • MSE: {mse:.4f}")
    logger.info(f"      • MAE: {mae:.4f}")
    logger.info(f"      • RMSE: {rmse:.4f}")
    logger.info(f"      • R²: {r2:.4f}")
    
    # Plot
    fig = model.plot(forecast)
    plt.title('Prophet Model Forecast')
    save_figure(plt.gcf(), 'prophet_forecast')
    plt.show()
    
    return model, predictions, metrics

@timer_decorator
def build_ensemble_model(data, forecast_days=30):
    """
    Build ensemble model combining multiple techniques
    """
    logger.info("\n🤝 Building Ensemble Model...")
    
    # Split data
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    # Model 1: Exponential Smoothing
    ses_model = ExponentialSmoothing(train, seasonal_periods=7,
                                     trend='add', seasonal='add')
    ses_fit = ses_model.fit()
    ses_pred = ses_fit.forecast(len(test))
    
    # Model 2: Random Forest with lag features
    def create_lag_features(data, lags=7):
        df_lag = pd.DataFrame(data)
        for i in range(1, lags+1):
            df_lag[f'lag_{i}'] = df_lag.shift(i)
        return df_lag.dropna()
    
    df_rf = create_lag_features(data)
    X = df_rf.drop(columns=[0]).values
    y = df_rf[0].values
    
    X_train, X_test = X[:train_size-7], X[train_size-7:train_size+len(test)-7]
    y_train, y_test = y[:train_size-7], y[train_size-7:train_size+len(test)-7]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Model 3: Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    
    # Dynamic weights based on performance
    weights = [0.25, 0.25, 0.25, 0.25]  # SES, ARIMA, RF, GB
    
    # Ensemble prediction
    ensemble_pred = (weights[0] * ses_pred[:len(test)] + 
                     weights[1] * rf_pred[:len(test)] + 
                     weights[2] * gb_pred[:len(test)])
    
    # Calculate metrics
    mse = mean_squared_error(test, ensemble_pred)
    mae = mean_absolute_error(test, ensemble_pred)
    r2 = r2_score(test, ensemble_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    logger.info(f"\n   📊 Ensemble Performance:")
    logger.info(f"      • MSE: {mse:.4f}")
    logger.info(f"      • MAE: {mae:.4f}")
    logger.info(f"      • RMSE: {rmse:.4f}")
    logger.info(f"      • R²: {r2:.4f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Time series plot
    axes[0,0].plot(test.index, test, label='Actual', linewidth=2)
    axes[0,0].plot(test.index, ensemble_pred, label='Ensemble', linewidth=2)
    axes[0,0].set_title('Ensemble Model: Actual vs Predicted')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Model comparison
    models = ['SES', 'RF', 'GB', 'Ensemble']
    scores = [
        mean_absolute_error(test, ses_pred[:len(test)]),
        mean_absolute_error(test, rf_pred[:len(test)]),
        mean_absolute_error(test, gb_pred[:len(test)]),
        mae
    ]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    axes[0,1].bar(models, scores, color=colors)
    axes[0,1].set_title('Model Comparison (MAE)')
    axes[0,1].set_ylabel('Mean Absolute Error')
    axes[0,1].grid(True, alpha=0.3)
    
    # Error distribution
    errors = test.values - ensemble_pred
    axes[1,0].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='#3B8F5E')
    axes[1,0].set_title('Ensemble Error Distribution')
    axes[1,0].set_xlabel('Prediction Error (°C)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1,1].scatter(test.values, ensemble_pred, alpha=0.5, s=10, color='#A23B72')
    axes[1,1].plot([test.min(), test.max()], [test.min(), test.max()], 'r--', lw=2)
    axes[1,1].set_title('Ensemble: Actual vs Predicted')
    axes[1,1].set_xlabel('Actual Temperature (°C)')
    axes[1,1].set_ylabel('Predicted Temperature (°C)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'ensemble_results')
    plt.show()
    
    models_dict = {
        'exponential_smoothing': ses_fit,
        'random_forest': rf_model,
        'gradient_boosting': gb_model
    }
    
    return ensemble_pred, metrics, models_dict

def compare_models(model_results):
    """
    Compare all models and create comparison table
    """
    comparison = pd.DataFrame(model_results).T
    comparison = comparison.round(4)
    comparison['ranking'] = comparison['r2'].rank(ascending=False).astype(int)
    comparison = comparison.sort_values('ranking')
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # R² comparison
    axes[0].barh(comparison.index, comparison['r2'], 
                 color=plt.cm.viridis(comparison['r2'] / comparison['r2'].max()))
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('Model Performance Comparison (R²)')
    axes[0].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[1].barh(comparison.index, comparison['mae'],
                 color=plt.cm.plasma(comparison['mae'] / comparison['mae'].max()))
    axes[1].set_xlabel('MAE')
    axes[1].set_title('Model Error Comparison (MAE)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(plt.gcf(), 'model_comparison')
    plt.show()
    
    return comparison