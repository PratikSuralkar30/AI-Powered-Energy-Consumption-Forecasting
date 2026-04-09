# AI-Powered Energy Consumption Forecasting System ⚡📈

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Flask](https://img.shields.io/badge/Flask-API-green)

An industry-oriented AI system built to forecast hourly electricity demand. This project simulates a real-world scenario (like a smart building or manufacturing plant) where energy consumption peaks and troughs based on the time of day and week.

## 🚀 The Problem Being Solved
Power grids and facilities often face **Unpredictable Energy Demand**. 
If a manufacturing plant exceeds its contracted energy peak, it faces massive penalties. If grid operators underestimate demand, blackouts occur.
This AI solves that by learning usage patterns and forecasting future needs, allowing for proactive, optimized energy management.

## 🧠 Architecture
- **Data:** Synthetic hourly smart-meter data representing 1 year of usage.
- **Machine Learning Model:** Multi-Layer Perceptron (MLP) Regressor (Scikit-Learn)
- **Deployment:** Real-time Flask API

[ Dataset ] -> [ Preprocessing (Fill Missing/Resample) ] -> [ Feature Engineering (Hour/Day Extraction) ] -> [ MLP Neural Network ] -> [ Flask API ]

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/AI-Energy-Consumption-Forecasting.git
   cd AI-Energy-Consumption-Forecasting
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Mac/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

### 1. Generate Data & Train the Model
First, generate the simulated energy data:
```bash
python simulate_data.py
```
*Creates `data/energy.csv`*

Next, train the AI model:
```bash
python src/train.py
```
*Evaluates the model, saves graphs to `outputs/`, and saves the model to `models/energy_forecast_model.pkl`*

### 2. Run the Real-Time Forecasting API
Start the Flask server so other applications can request forecasts:
```bash
python main.py
```
*API runs on `http://127.0.0.1:5000`*

### 3. Test the API
In another terminal, you can test the API by asking it to forecast energy for a specific hour and day.

**Using cURL:**
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"hour": 14, "day": 2}'
```

**Expected JSON Response:**
```json
{
  "input_day": 2,
  "input_hour": 14,
  "predicted_energy_kw": 2540.32
}
```

## 📊 Results Summary
- **Mean Absolute Error (MAE):** ~300 kW (varies by generation)
- The model successfully learned that energy spikes dramatically between 8 AM and 6 PM on weekdays, while staying flat on weekends.

### Screenshots
*(Add your screenshots here via the `images/` folder)*
![Actual vs Predicted](outputs/actual_vs_predicted.png)
*(Note: Upload these to your images/ folder and adjust the link)*

---

This project was built to demonstrate proficiency in Data Engineering, Machine Learning Pipeline Creation, and API Deployment.
