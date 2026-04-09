import pandas as pd
import os

def load_and_clean_data(filepath):
    print("Loading dataset...")
    # Parse dates and set as index
    data = pd.read_csv(filepath, parse_dates=['Datetime'], index_col='Datetime')
    
    # Resample to hourly mean to ensure consistent time steps
    data = data.resample('H').mean()
    
    # Fill missing values using forward fill
    data = data.ffill()
    
    # Feature Engineering
    data['hour'] = data.index.hour
    data['day'] = data.index.dayofweek
    
    # Drop rows with NaN
    data = data.dropna()
    return data

if __name__ == "__main__":
    # Assuming the user places energy.csv in the data folder
    df = load_and_clean_data('../data/energy.csv')
    df.to_csv('../data/processed_energy.csv')
    print("Data preprocessed and saved successfully.")
