#!/usr/bin/env python
"""
Runner script for Global Weather Forecasting Project
"""

import os
import sys
import subprocess
import webbrowser
import src.main

def check_requirements():
    """Check if requirements are installed"""
    try:
        import pandas
        import numpy
        import sklearn
        import tensorflow
        import plotly
        import matplotlib
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'outputs/figures', 'outputs/models', 'notebooks']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_data_file():
    """Check if data file exists"""
    data_path = 'data/Global Weather Repository.csv'
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
        print(f"✅ Data file found: {data_path} ({file_size:.2f} MB)")
        return True
    else:
        print(f"❌ Data file not found at {data_path}")
        print("Please download the dataset from Kaggle and place it in the data/ folder")
        return False

def run_analysis():
    """Run the main analysis"""
    print("\n" + "="*60)
    print("Running Global Weather Forecasting Analysis")
    print("="*60 + "\n")
    
    # Run the main script
    result = subprocess.run([sys.executable, 'src/main.py'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print("\n✅ Analysis completed successfully!")
        return True
    else:
        print("\n❌ Analysis failed with error:")
        print(result.stderr)
        return False

def open_outputs():
    """Open output directory"""
    outputs_path = os.path.abspath('outputs')
    if os.path.exists(outputs_path):
        webbrowser.open(f'file://{outputs_path}')
        print(f"📁 Outputs available at: {outputs_path}")

def main():
    """Main runner function"""
    print("\n" + "="*60)
    print("🌤️  Global Weather Forecasting Project Runner")
    print("="*60 + "\n")
    
    # Check requirements
    if not check_requirements():
        return
    
    # Setup directories
    setup_directories()
    
    # Check data file
    if not check_data_file():
        return
    
    # Ask user what to do
    print("\n" + "="*60)
    print("What would you like to do?")
    print("1. Run complete analysis")
    print("2. Check only (don't run analysis)")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        if run_analysis():
            open_outputs()
    elif choice == '2':
        print("\n✅ System check completed. Ready to run analysis.")
    else:
        print("\nExiting...")
    
    print("\n" + "="*60)
    print("Thank you for using the Weather Forecasting Project!")
    print("="*60)

if __name__ == "__main__":
    main()