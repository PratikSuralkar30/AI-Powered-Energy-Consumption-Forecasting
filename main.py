from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model once when the app starts
model_path = 'models/energy_forecast_model.pkl'
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure to run 'python src/train.py' first to generate the model!")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "AI Energy Forecasting System API is Running!",
        "endpoints": {
            "/predict [POST]": "Pass JSON with 'hour' (0-23) and 'day' (0-6) to get prediction."
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from user
        data = request.get_json()
        
        hour = data.get('hour')
        day = data.get('day')
        
        if hour is None or day is None:
            return jsonify({'error': 'Please provide both hour and day parameters'}), 400
            
        # Format for sklearn
        features = np.array([[hour, day]])
        
        # Predict using AI model
        prediction = model.predict(features)[0]
        
        return jsonify({
            'input_hour': hour,
            'input_day': day,
            'predicted_energy_kw': round(prediction, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting AI Server...")
    app.run(debug=True, port=5000)
