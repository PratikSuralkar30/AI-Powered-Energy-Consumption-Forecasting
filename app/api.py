from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('../models/energy_forecast_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        hour = data['hour']
        day = data['day']
        
        features = np.array([[hour, day]])
        prediction = model.predict(features)
        
        return jsonify({
            'status': 'success',
            'input': {'hour': hour, 'day': day},
            'predicted_energy_MW': round(prediction[0], 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
