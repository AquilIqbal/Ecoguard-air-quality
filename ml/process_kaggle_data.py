"""
Kaggle Dataset Processor
========================

Processes the Kaggle India Air Quality dataset (city_day.csv)
and converts it to our training format.

Dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
Source: Central Pollution Control Board (CPCB), India

Author: EcoGuard Team
Date: February 2026
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def load_kaggle_data(filepath='../data/city_day.csv'):
    """
    Load Kaggle India Air Quality dataset
    
    Args:
        filepath: Path to city_day.csv
        
    Returns:
        DataFrame with raw data
    """
    print("=" * 60)
    print("📂 LOADING KAGGLE DATASET")
    print("=" * 60)
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print("   Download from: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india")
        raise FileNotFoundError(f"city_day.csv not found at {filepath}")
    
    print(f"Loading: {filepath}")
    df = pd.read_csv(filepath)
    
    print(f"✅ Loaded {len(df)} rows")
    print(f"📊 Columns: {list(df.columns)}")
    print(f"🏙️  Cities: {df['City'].nunique()}")
    print(f"📅 Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def clean_and_prepare(df):
    """
    Clean and prepare Kaggle data
    
    Args:
        df: Raw Kaggle DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\n" + "=" * 60)
    print("🔧 CLEANING AND PREPARING DATA")
    print("=" * 60)
    
    df = df.copy()
    
    # Rename columns to lowercase and standardize
    df.columns = df.columns.str.lower()
    
    # Rename columns to match our format
    column_mapping = {
        'city': 'city',
        'date': 'date',
        'pm2.5': 'pm2_5',
        'pm10': 'rspm',  # Using PM10 as RSPM
        'no2': 'no2',
        'so2': 'so2',
        'aqi': 'aqi'
    }
    
    # Select and rename columns
    available_cols = [col for col in column_mapping.keys() if col in df.columns]
    df = df[available_cols].rename(columns=column_mapping)
    
    print(f"✓ Renamed columns to our format")
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    print(f"✓ Converted dates")
    
    # Handle missing values
    initial_rows = len(df)
    
    # Fill missing pollutant values with forward fill, then backward fill
    pollutant_cols = ['pm2_5', 'rspm', 'no2', 'so2']
    for col in pollutant_cols:
        if col in df.columns:
            df[col] = df.groupby('city')[col].fillna(method='ffill').fillna(method='bfill')
    
    # Drop rows still with NaN in critical columns
    df = df.dropna(subset=['pm2_5', 'rspm', 'date', 'city'])
    
    print(f"✓ Handled missing values (dropped {initial_rows - len(df)} rows)")
    
    # Remove outliers (values > 999 are likely errors)
    for col in pollutant_cols:
        if col in df.columns:
            df = df[df[col] < 999]
    
    print(f"✓ Removed outliers")
    
    # Calculate AQI if not present
    if 'aqi' not in df.columns or df['aqi'].isna().sum() > len(df) * 0.5:
        print("✓ Calculating AQI from pollutants...")
        df['aqi'] = (
            0.4 * df['pm2_5'] +
            0.3 * df['rspm'] +
            0.2 * df['no2'].fillna(0) +
            0.1 * df['so2'].fillna(0)
        )
    
    # Sort by city and date
    df = df.sort_values(['city', 'date'])
    
    print(f"✓ Sorted by city and date")
    
    return df


def filter_major_cities(df, min_records=500):
    """
    Filter to keep only cities with sufficient data
    
    Args:
        df: Cleaned DataFrame
        min_records: Minimum number of records required per city
        
    Returns:
        Filtered DataFrame
    """
    print("\n" + "=" * 60)
    print("🏙️  FILTERING CITIES")
    print("=" * 60)
    
    # Count records per city
    city_counts = df.groupby('city').size()
    
    print(f"\nRecords per city:")
    print(city_counts.sort_values(ascending=False).head(20))
    
    # Filter cities with enough data
    valid_cities = city_counts[city_counts >= min_records].index.tolist()
    
    df_filtered = df[df['city'].isin(valid_cities)]
    
    print(f"\n✅ Kept {len(valid_cities)} cities with ≥{min_records} records")
    print(f"   Cities: {', '.join(sorted(valid_cities))}")
    print(f"   Total records: {len(df_filtered)}")
    
    return df_filtered


def save_processed_data(df, output_path='../data/kaggle_processed.csv'):
    """
    Save processed data
    
    Args:
        df: Processed DataFrame
        output_path: Output file path
    """
    print("\n" + "=" * 60)
    print("💾 SAVING PROCESSED DATA")
    print("=" * 60)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save
    df.to_csv(output_path, index=False)
    
    print(f"✅ Saved to: {output_path}")
    print(f"📁 Size: {os.path.getsize(output_path) / 1024:.2f} KB")
    print(f"📊 Records: {len(df)}")
    print(f"🏙️  Cities: {df['city'].nunique()}")
    print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")


def show_summary(df):
    """
    Show dataset summary
    
    Args:
        df: Processed DataFrame
    """
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    
    # Summary by city
    summary = df.groupby('city').agg({
        'aqi': ['count', 'mean', 'min', 'max'],
        'pm2_5': 'mean',
        'rspm': 'mean'
    }).round(2)
    
    print("\n" + summary.to_string())
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("📈 OVERALL STATISTICS")
    print("=" * 60)
    print(f"Total records: {len(df)}")
    print(f"Cities: {df['city'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Days of data: {(df['date'].max() - df['date'].min()).days}")
    print(f"\nAverage AQI: {df['aqi'].mean():.2f}")
    print(f"Average PM2.5: {df['pm2_5'].mean():.2f}")
    print(f"Average PM10: {df['rspm'].mean():.2f}")


def main():
    """
    Main processing pipeline
    """
    print("\n" + "🌿 " + "=" * 56 + " 🌿")
    print("   ECOGUARD - KAGGLE DATA PROCESSOR")
    print("🌿 " + "=" * 56 + " 🌿\n")
    
    # Load data
    df = load_kaggle_data()
    
    # Clean and prepare
    df_clean = clean_and_prepare(df)
    
    # Filter cities (keep those with ≥500 records)
    df_filtered = filter_major_cities(df_clean, min_records=500)
    
    # Save processed data
    save_processed_data(df_filtered, output_path='../data/kaggle_processed.csv')
    
    # Show summary
    show_summary(df_filtered)
    
    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Dataset ready: data/kaggle_processed.csv")
    print("   2. Update training scripts to use this file")
    print("   3. Run: python train_all_models.py")
    print("   4. Run: python train_global_model.py")
    print("\n🎯 You now have REAL government data for training!")


if __name__ == "__main__":
    main()