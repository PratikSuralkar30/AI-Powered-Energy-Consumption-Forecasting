import pandas as pd
import numpy as np
import os

def generate_synthetic_data(num_days=365):
    """
    Generates synthetic hourly energy consumption data.
    Simulates a smart meter reading for a typical office building.
    - Peaks during daytime (9 AM - 6 PM).
    - Lower usage at night.
    - Lower usage on weekends.
    - Random noise added for realism.
    """
    print(f"Generating {num_days} days of synthetic energy data...")
    
    # Create hourly datetime index
    dates = pd.date_range(start="2023-01-01", periods=num_days*24, freq="h")
    
    # Base load (minimum energy used e.g. servers, security running)
    base_load = 500  
    
    energy = []
    for dt in dates:
        hour = dt.hour
        day = dt.dayofweek
        
        # Weekend logic (Day 5 is Saturday, Day 6 is Sunday)
        if day >= 5:
            hourly_usage = base_load + np.random.normal(100, 50)
        else:
            # Weekday logic
            if 8 <= hour <= 18:
                # Working hours peak
                hourly_usage = base_load + 2000 + np.random.normal(500, 200)
            else:
                # Nighttime
                hourly_usage = base_load + np.random.normal(200, 100)
                
        energy.append(hourly_usage)
        
    df = pd.DataFrame({'Datetime': dates, 'Energy': energy})
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    file_path = 'data/energy.csv'
    df.to_csv(file_path, index=False)
    print(f"Successfully generated data and saved to {file_path}")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data()
