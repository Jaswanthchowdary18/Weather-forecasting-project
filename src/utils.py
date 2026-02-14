"""
Utility functions for Weather Forecasting Project
PROFESSIONAL VERSION with comprehensive utilities
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import pickle
from datetime import datetime
from pathlib import Path

# Configure logging
def setup_logging(log_level=logging.INFO):
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    log_filename = f"logs/weather_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

def print_pm_mission():
    """Print PM Accelerator mission with styling"""
    mission = """
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                    🚀 PM ACCELERATOR MISSION                                ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║  Empowering organizations to achieve excellence through AI-driven solutions ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    print(mission)

def create_output_directories():
    """Create all necessary output directories"""
    directories = [
        'outputs/figures',
        'outputs/models',
        'outputs/reports',
        'outputs/data',
        'logs',
        'temp'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"✅ Directory ensured: {directory}")

def reduce_memory_usage(df, verbose=True):
    """
    Reduce memory usage of dataframe by downcasting numeric types
    """
    if verbose:
        logger.info("Optimizing memory usage...")
        start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        reduction = 100 * (start_mem - end_mem) / start_mem
        logger.info(f"Memory usage reduced: {start_mem:.2f} MB → {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df

def save_figure(fig, filename, subdir='figures', dpi=300):
    """Save figure to specified directory with multiple formats"""
    directory = f'outputs/{subdir}'
    os.makedirs(directory, exist_ok=True)
    
    # Save as PNG
    png_path = os.path.join(directory, f"{filename}.png")
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    # Save as PDF (vector format)
    pdf_path = os.path.join(directory, f"{filename}.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    
    logger.debug(f"Figure saved: {png_path}")
    return png_path

def save_model(model, filename, subdir='models', method='joblib'):
    """Save trained model using specified method"""
    directory = f'outputs/{subdir}'
    os.makedirs(directory, exist_ok=True)
    
    if method == 'joblib':
        path = os.path.join(directory, f"{filename}.joblib")
        joblib.dump(model, path, compress=3)
    elif method == 'pickle':
        path = os.path.join(directory, f"{filename}.pkl")
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    elif method == 'keras':
        path = os.path.join(directory, f"{filename}.h5")
        model.save(path)
    
    logger.info(f"✅ Model saved: {path}")
    return path

def load_model(filename, subdir='models', method='joblib'):
    """Load trained model"""
    directory = f'outputs/{subdir}'
    
    if method == 'joblib':
        path = os.path.join(directory, f"{filename}.joblib")
        model = joblib.load(path)
    elif method == 'pickle':
        path = os.path.join(directory, f"{filename}.pkl")
        with open(path, 'rb') as f:
            model = pickle.load(f)
    
    logger.info(f"✅ Model loaded: {path}")
    return model

def validate_data_quality(df):
    """Comprehensive data quality validation"""
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'data_types': df.dtypes.astype(str).to_dict(),
        'memory_usage_mb': df.memory_usage().sum() / 1024**2,
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns)
    }
    
    # Log quality issues
    logger.info("📊 Data Quality Report:")
    logger.info(f"  • Total Records: {report['total_rows']:,}")
    logger.info(f"  • Total Features: {report['total_columns']}")
    logger.info(f"  • Duplicate Rows: {report['duplicates']}")
    logger.info(f"  • Memory Usage: {report['memory_usage_mb']:.2f} MB")
    
    # Check for columns with high missing values
    high_missing = {k: v for k, v in report['missing_percentage'].items() if v > 20}
    if high_missing:
        logger.warning(f"  ⚠️ Columns with >20% missing: {list(high_missing.keys())}")
    
    return report

def save_results(results, filename):
    """Save analysis results to JSON"""
    path = f'outputs/reports/{filename}.json'
    
    # Convert non-serializable types
    def json_serializer(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                           np.int16, np.int32, np.int64, np.uint8,
                           np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, default=json_serializer, indent=2)
    
    logger.info(f"✅ Results saved: {path}")

def setup_visualization_style():
    """Set consistent visualization style"""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Custom color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B8F5E']
    
    # Update rcParams
    plt.rcParams.update({
        'figure.figsize': (12, 6),
        'figure.dpi': 100,
        'figure.facecolor': 'white',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'lines.linewidth': 2,
        'lines.markersize': 8,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })
    
    return colors

def timer_decorator(func):
    """Decorator to time function execution"""
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        duration = end - start
        logger.info(f"⏱️ {func.__name__} executed in {duration}")
        return result
    return wrapper