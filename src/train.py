import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

def train_model():
    print("Loading datasets...")
    # Load the Data
    # parse_dates connects Pandas to understand timestamps
    # index_col sets the Datetime column as the structural index of the DataFrame
    data = pd.read_csv('data/energy.csv', parse_dates=['Datetime'], index_col='Datetime')
    
    # Preprocessing
    print("Preprocessing data...")
    # Resample to Hourly means 'H' ensuring there are no missing hours
    data = data.resample('h').mean()
    # ffill forwards fills missing data
    data = data.ffill() 

    # Feature Engineering
    # The AI does not understand raw dates, it understands numerical patterns
    print("Engineering features...")
    data['hour'] = data.index.hour
    data['day'] = data.index.dayofweek
    
    # Features (X) and Target (y)
    X = data[['hour', 'day']]
    y = data['Energy']

    # Train/Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Building using Multi-Layer Perceptron Regressor
    print("Training AI Model (MLP Regressor)...")
    # Hidden layers defines the 'brain' structure. 
    model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    model.fit(X_train, y_train)

    print("Model Trained! Generating Predictions...")
    predictions = model.predict(X_test)
    
    # Evaluation Metrics
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Mean Absolute Error (MAE): {mae:.2f} kW")
    print(f"R-squared Score (Accuracy): {r2:.2f}")

    # Output Visualization
    print("Generating visualizations...")
    os.makedirs('outputs', exist_ok=True)
    
    plt.figure(figsize=(15, 6))
    # Let's plot just a subset (first 100 predictions) to make it readable
    plt.plot(y_test.values[:100], label='Actual Energy Usage', marker='o')
    plt.plot(predictions[:100], label='Predicted AI Usage', marker='x', linestyle='--')
    plt.title('AI Forecast vs Actual Energy Usage (Sample)')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Energy Consumption (kW)')
    plt.legend()
    plt.grid(True)
    plt.savefig('outputs/actual_vs_predicted.png')
    print("Graph saved to outputs/actual_vs_predicted.png")

    # Save Model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/energy_forecast_model.pkl')
    print("AI Model Saved to models/energy_forecast_model.pkl")

if __name__ == "__main__":
    train_model()
