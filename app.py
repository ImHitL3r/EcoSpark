from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


MODEL_DIR = 'model'
MODEL_FILENAME = 'solar_wind_predictor_nasa_model_100loc.joblib'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)


print(f"Loading model from {MODEL_PATH}...")
model_dict = None
model_features = ['latitude', 'longitude', 'day_of_year']
solar_model = None
wind_model = None
model_evaluation = None

try:
    if os.path.exists(MODEL_PATH):
        model_dict = joblib.load(MODEL_PATH)
        solar_model = model_dict.get('solar_model')
        wind_model = model_dict.get('wind_model')
        model_features = model_dict.get('features', model_features)
        model_evaluation = model_dict.get('evaluation')

        if solar_model and wind_model:
            print("Model loaded successfully.")
            print(f"Model expects features: {model_features}")
            if model_evaluation:
                print(f"Stored Evaluation Metrics: {model_evaluation}")
        else:
            print("ERROR: Model file loaded but seems incomplete.")
            model_dict = None; solar_model = None; wind_model = None
    else:
        print(f"ERROR: Model file not found at {MODEL_PATH}.")
        print(f"Please run the corresponding model_training.py script first to create '{MODEL_FILENAME}'.")

except Exception as e:
    print(f"An unexpected error occurred loading the model: {e}")
    model_dict = None; solar_model = None; wind_model = None

#Flask App
app = Flask(__name__)

#Thresholds
SOLAR_THRESHOLDS = {'good': 5.0, 'moderate': 4.0} #kWh/m²/day
WIND_THRESHOLDS = {'good': 6.0, 'moderate': 4.5} #m/s @ 10m


def get_assessment(value, thresholds):
    """Provides a simple assessment based on value and thresholds."""
    if value is None:
        return "N/A"
    if value >= thresholds['good']:
        return "Good"
    elif value >= thresholds['moderate']:
        return "Moderate"
    else:
        return "Poor"


@app.route('/')
def index():
    """Renders the main page."""
    print("Serving index.html")
    return render_template('index.html',
                           initial_lat=39.8283,
                           initial_lon=-98.5795,
                           initial_zoom=4)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests for TOMORROW using the loaded model."""
    print("Received prediction request for tomorrow.")

    if not solar_model or not wind_model:
        print("Error: Model not available.")
        return jsonify({'error': 'Model not available on server.'}), 500

    try:
        req_data = request.get_json()
        print(f"Request data: {req_data}")

        if not req_data or 'latitude' not in req_data or 'longitude' not in req_data:
            print("Error: Missing latitude or longitude.")
            return jsonify({'error': 'Missing latitude or longitude'}), 400

        latitude = float(req_data['latitude'])
        longitude = float(req_data['longitude'])

        
        tomorrow_date = datetime.now() + timedelta(days=1)
        tomorrow_day_of_year = tomorrow_date.timetuple().tm_yday
        tomorrow_date_str = tomorrow_date.strftime('%Y-%m-%d')
        print(f"Using tomorrow's day of year for prediction: {tomorrow_day_of_year} ({tomorrow_date_str})")

        
        input_values = {'latitude': latitude, 'longitude': longitude, 'day_of_year': tomorrow_day_of_year}
        input_data = pd.DataFrame([input_values], columns=model_features)

        print(f"Prepared input data for model:\n{input_data}")

        #Make Predictions
        solar_prediction = solar_model.predict(input_data)[0]
        wind_prediction = wind_model.predict(input_data)[0]
        solar_prediction = max(0, round(solar_prediction, 2))
        wind_prediction = max(0, round(wind_prediction, 2))
        print(f"Prediction - Solar: {solar_prediction} kWh/m²/day, Wind: {wind_prediction} m/s")

        
        solar_assessment = get_assessment(solar_prediction, SOLAR_THRESHOLDS)
        wind_assessment = get_assessment(wind_prediction, WIND_THRESHOLDS)
        print(f"Assessment - Solar: {solar_assessment}, Wind: {wind_assessment}")

        
        result = {
            'latitude': latitude,
            'longitude': longitude,
            'predicted_solar_radiation': solar_prediction,
            'predicted_wind_speed': wind_prediction,
            'solar_assessment': solar_assessment,
            'wind_assessment': wind_assessment,
            'prediction_date': tomorrow_date_str,
            'prediction_day_of_year': tomorrow_day_of_year,
            'data_source': 'ML Model (Trained on NASA POWER Historical Data)'
        }
        print(f"Prepared result: {result}")

        return jsonify(result)

    except ValueError:
        print("Error: Invalid latitude or longitude format.")
        return jsonify({'error': 'Invalid latitude or longitude format'}), 400
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return jsonify({'error': 'Prediction failed due to an internal server error'}), 500

#Run the App
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
         print("\n--- WARNING ---")
         print(f"Model file '{MODEL_PATH}' not found.")
         print("Please run the corresponding 'model_training.py' script first.")
         print("---------------\n")
        

    app.run(debug=True, host='0.0.0.0', port=5000)

