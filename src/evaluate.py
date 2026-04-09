import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

def evaluate():
    model = joblib.load('../models/energy_forecast_model.pkl')
    X_test = pd.read_csv('../data/X_test.csv')
    y_test = pd.read_csv('../data/y_test.csv')
    
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")
    
    plt.figure(figsize=(15, 5))
    plt.plot(y_test.values[:100], label='Actual Usage', color='blue')
    plt.plot(predictions[:100], label='Predicted Usage', color='orange', linestyle='dashed')
    plt.title('Energy Consumption: Actual vs Predicted')
    plt.xlabel('Hours')
    plt.ylabel('Energy (MW)')
    plt.legend()
    
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/forecast_comparison.png')
    print("Graph saved to outputs/forecast_comparison.png")

if __name__ == "__main__":
    evaluate()
