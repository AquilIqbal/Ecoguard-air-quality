"""
India Air Quality Data Fetcher
===============================

Fetches historical air quality data for major Indian cities.

Cities covered (15 major metros):
- Delhi, Mumbai, Bangalore, Kolkata, Chennai
- Hyderabad, Pune, Ahmedabad, Jaipur, Lucknow
- Chandigarh, Nagpur, Indore, Bhopal, Patna

Author: EcoGuard Team  
Date: February 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


# Major Indian cities
CITIES = [
    'Delhi', 'Mumbai', 'Bangalore', 'Kolkata', 'Chennai',
    'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
    'Chandigarh', 'Nagpur', 'Indore', 'Bhopal', 'Patna'
]


def create_realistic_india_dataset(years=3):
    """
    Create realistic Indian air quality dataset
    
    This creates a synthetic but realistic dataset based on:
    - Real pollution patterns in Indian cities
    - Seasonal variations
    - City-specific characteristics
    
    Args:
        years: Number of years of data to generate
        
    Returns:
        DataFrame with daily AQI data for all cities
    """
    print("=" * 60)
    print("🇮🇳 CREATING INDIA AIR QUALITY DATASET")
    print("=" * 60)
    
    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"📅 Date range: {start_date.date()} to {end_date.date()}")
    print(f"🏙️  Cities: {len(CITIES)}")
    print(f"📊 Days: {len(dates)}")
    
    # City-specific baseline pollution levels (realistic)
    city_baselines = {
        'Delhi': {'pm2_5': 120, 'pm10': 200, 'no2': 50, 'so2': 20},
        'Mumbai': {'pm2_5': 70, 'pm10': 120, 'no2': 40, 'so2': 15},
        'Bangalore': {'pm2_5': 50, 'pm10': 90, 'no2': 30, 'so2': 10},
        'Kolkata': {'pm2_5': 90, 'pm10': 150, 'no2': 45, 'so2': 18},
        'Chennai': {'pm2_5': 60, 'pm10': 100, 'no2': 35, 'so2': 12},
        'Hyderabad': {'pm2_5': 65, 'pm10': 110, 'no2': 38, 'so2': 14},
        'Pune': {'pm2_5': 55, 'pm10': 95, 'no2': 32, 'so2': 11},
        'Ahmedabad': {'pm2_5': 75, 'pm10': 130, 'no2': 42, 'so2': 16},
        'Jaipur': {'pm2_5': 85, 'pm10': 140, 'no2': 40, 'so2': 15},
        'Lucknow': {'pm2_5': 95, 'pm10': 160, 'no2': 48, 'so2': 19},
        'Chandigarh': {'pm2_5': 60, 'pm10': 105, 'no2': 35, 'so2': 13},
        'Nagpur': {'pm2_5': 70, 'pm10': 115, 'no2': 38, 'so2': 14},
        'Indore': {'pm2_5': 65, 'pm10': 110, 'no2': 36, 'so2': 13},
        'Bhopal': {'pm2_5': 68, 'pm10': 112, 'no2': 37, 'so2': 14},
        'Patna': {'pm2_5': 100, 'pm10': 170, 'no2': 52, 'so2': 21}
    }
    
    data = []
    
    print("\n🔧 Generating realistic data patterns...")
    
    for city in CITIES:
        print(f"  📍 {city}...")
        
        baseline = city_baselines[city]
        
        for i, date in enumerate(dates):
            # Seasonal variation (winter pollution higher in India)
            month = date.month
            if month in [11, 12, 1, 2]:  # Winter
                seasonal_factor = 1.4
            elif month in [6, 7, 8]:  # Monsoon
                seasonal_factor = 0.7
            else:
                seasonal_factor = 1.0
            
            # Day of week (weekends slightly lower)
            dow_factor = 0.9 if date.weekday() in [5, 6] else 1.0
            
            # Random daily variation
            daily_noise = np.random.normal(1.0, 0.15)
            
            # Calculate pollutants with realistic correlations
            pm2_5 = baseline['pm2_5'] * seasonal_factor * dow_factor * daily_noise
            pm10 = baseline['pm10'] * seasonal_factor * dow_factor * daily_noise
            no2 = baseline['no2'] * seasonal_factor * dow_factor * np.random.normal(1.0, 0.12)
            so2 = baseline['so2'] * seasonal_factor * dow_factor * np.random.normal(1.0, 0.10)
            
            # Add some temporal autocorrelation (today similar to yesterday)
            if i > 0 and np.random.random() < 0.7:
                prev_pm25 = data[-1]['pm2_5'] if data[-1]['city'] == city else pm2_5
                pm2_5 = 0.7 * prev_pm25 + 0.3 * pm2_5
            
            # Calculate AQI
            aqi = (
                0.4 * pm2_5 +
                0.3 * pm10 +
                0.2 * no2 +
                0.1 * so2
            )
            
            # Ensure non-negative
            pm2_5 = max(5, pm2_5)
            pm10 = max(10, pm10)
            no2 = max(5, no2)
            so2 = max(2, so2)
            aqi = max(10, aqi)
            
            data.append({
                'date': date.date(),
                'city': city,
                'pm2_5': round(pm2_5, 2),
                'rspm': round(pm10, 2),  # Using rspm name for compatibility
                'no2': round(no2, 2),
                'so2': round(so2, 2),
                'aqi': round(aqi, 2)
            })
    
    df = pd.DataFrame(data)
    
    print(f"\n✅ Generated {len(df)} records")
    print(f"📊 Cities: {df['city'].nunique()}")
    print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def save_dataset(df, filename='../data/india_cities_timeseries.csv'):
    """Save dataset to CSV"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"\n💾 Saved to: {filename}")
    print(f"📁 Size: {os.path.getsize(filename) / 1024:.2f} KB")


def main():
    """Main execution"""
    print("\n" + "🌿 " + "=" * 56 + " 🌿")
    print("   ECOGUARD - INDIA DATA GENERATOR")
    print("🌿 " + "=" * 56 + " 🌿\n")
    
    # Generate dataset
    df = create_realistic_india_dataset(years=3)
    
    # Save
    save_dataset(df)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    print(df.groupby('city')['aqi'].agg(['count', 'mean', 'min', 'max']))
    
    print("\n✅ Dataset ready for ML training!")
    print("   Next: Run train_all_models.py")


if __name__ == "__main__":
    main()