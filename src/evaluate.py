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
    
    # Professional Dark Mode visualization
    plt.style.use('dark_background')
    plt.figure(figsize=(12, 6), dpi=150)
    
    # Plotting with smoother lines and better colors
    # Ensure y_test values are flattened for fill_between
    y_vals = y_test.values.flatten() if hasattr(y_test, 'values') else y_test
    
    plt.plot(y_vals[:100], label='Actual Demand', color='#00f2ff', linewidth=2, alpha=0.8)
    plt.plot(predictions[:100], label='AI Prediction', color='#ff007b', linewidth=2, linestyle='--')
    
    # Adding a glow effect (subtle)
    plt.fill_between(range(100), predictions[:100], y_vals[:100], color='#00f2ff', alpha=0.1)
    
    plt.title('AI Energy Forecasting: Actual vs. Predicted Load', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Time (Hours)', fontsize=12)
    plt.ylabel('Energy Consumption (MW)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.legend(frameon=True, facecolor='black', edgecolor='white')
    
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/linkedin_forecast.png', bbox_inches='tight', transparent=False)
    print("Graph saved to outputs/linkedin_forecast.png")

if __name__ == "__main__":
    evaluate()
