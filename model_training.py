import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import numpy as np
import os
import time

print("Starting model training & evaluation process using NASA POWER historical data...")
print("Using expanded location list (~100 locations). This will take a VERY long time.")

#ACTUAL NASA POWER API KEY !!!
NASA_API_KEY = "tODJDMG2tk5VrFqsuHvJbkWQaTqAbIAJ6ACIPs0I"

# Model saving configuration
MODEL_DIR = 'model'
MODEL_FILENAME = 'solar_wind_predictor_nasa_model_100loc.joblib'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

#Feature and Target Definitions
FEATURES = ['latitude', 'longitude', 'day_of_year']
TARGETS = ['ALLSKY_SFC_SW_DWN', 'WS10M']

BASE_API_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START_DATE_TRAIN = "20180101" 
END_DATE_TRAIN = "20221231"  
PARAMETERS_TRAIN = ",".join(TARGETS)
COMMUNITY = "RE"
FORMAT = "JSON"

#~100 locations
SAMPLE_LOCATIONS_TRAIN = [

    (40.71, -74.00), (42.36, -71.06), (43.66, -70.26), (44.95, -68.77), (41.82, -71.41),
    (39.95, -75.16), (38.90, -77.04), (39.29, -76.61), (37.54, -77.43), (33.75, -84.39),
    (35.23, -80.84), (25.76, -80.19), (27.95, -82.46), (30.33, -81.66), (41.88, -87.63),
    (44.98, -93.27), (41.25, -95.93), (39.10, -94.58), (41.59, -93.62), (43.07, -89.40),
    (29.76, -95.36), (30.27, -97.74), (32.77, -96.80), (34.75, -92.29), (35.47, -97.52),
    (39.74, -104.99), (33.45, -112.07), (40.76, -111.89), (45.52, -111.04), (43.61, -116.20),
    (34.05, -118.24), (47.61, -122.33), (37.77, -122.42), (32.72, -117.16), (45.51, -122.68),
    
    (43.07, -70.82), (44.26, -72.57), (41.76, -72.68), 
    (36.85, -75.98), (40.44, -79.99), 
    (36.17, -86.78), (35.15, -90.05), (34.00, -81.03), 
    (29.42, -98.49), (31.76, -106.49), 
    (36.11, -115.17), 
    (34.00, -106.65), (35.08, -106.65), 
    (48.76, -122.47), (47.45, -122.31), 
    (46.88, -114.00), (46.60, -112.04), 
    (44.06, -121.31), (42.37, -122.87), 
    (39.32, -120.18), (36.60, -121.89), 
    (38.58, -121.49), (37.34, -121.89), 
    (33.68, -117.83), (33.95, -118.40), 
    (61.22, -149.90), (64.84, -147.72), 
    (21.31, -157.86), (20.89, -156.44), 
    (40.02, -83.00), (39.10, -84.51), 
    (42.33, -83.05), (42.73, -84.55), 
    (39.77, -86.16), (41.68, -86.25), 
    (38.25, -85.76), (37.99, -87.57), 
    (32.29, -90.18), (30.45, -91.19), 
    (35.96, -78.64), (34.21, -77.88), 
    (32.08, -81.10), 
    (29.95, -90.08), 
    (32.32, -86.30), 
    (45.03, -93.99), 
    (40.81, -96.70), 
    (38.63, -90.20), 
    (36.15, -95.99), 
    (31.55, -97.15), 
    (39.05, -108.59), 
    (40.59, -105.09), 
    (44.56, -104.71), (42.87, -106.31), 
    (40.58, -122.39), 
    (39.53, -119.81), 
    (44.05, -103.23), 
    (46.81, -100.78), 
    (46.87, -96.79),  
    (40.00, -76.30),  
    (42.10, -72.59),  
    (43.00, -76.15)   
]


#Fetch Historical Data
def fetch_nasa_power_historical_data(latitude, longitude, api_key):
    """Fetches daily solar and wind data for a location over a date range."""
    api_params = {
        "start": START_DATE_TRAIN, "end": END_DATE_TRAIN, "latitude": latitude,
        "longitude": longitude, "community": COMMUNITY, "parameters": PARAMETERS_TRAIN,
        "format": FORMAT, "header": "true", "api_key": api_key
    }
    response = None
    try:
        print(f"Fetching data for Lat: {latitude:.2f}, Lon: {longitude:.2f}...", end=' ')
        response = requests.get(BASE_API_URL, params=api_params, timeout=120)
        response.raise_for_status() #to avoid 422 error
        data = response.json()
        if "properties" not in data or "parameter" not in data["properties"]:
             print("Warning: Unexpected API response format. Skipping.")
             return None
        params_data = data["properties"]["parameter"]
        if not params_data:
             print("Warning: No parameter data returned. Skipping.")
             return None
        df = pd.DataFrame(params_data)
        if df.empty or not all(d.isdigit() and len(d) == 8 for d in df.index):
             print("Warning: Invalid data structure. Skipping.")
             return None
        df.index = pd.to_datetime(df.index, format='%Y%m%d')
        df['latitude'] = latitude
        df['longitude'] = longitude
        df.replace(-999, np.nan, inplace=True)
        print(f"Success ({len(df)} records).")
        return df
    except requests.exceptions.RequestException as e:
        status_code = response.status_code if response is not None else "N/A"
        print(f"Error: {e} (Status: {status_code})")
    
        return None
    except Exception as e:
        print(f"An unexpected error occurred processing data: {e}")
        return None

#Main Data Acquisition Loop
all_data_frames = []
if NASA_API_KEY == "PASTE_YOUR_NASA_POWER_API_KEY_HERE":
    print("\n--- ERROR: Please set your NASA POWER API key in the script. ---")
    exit()
else:
    print(f"\nFetching data for {len(SAMPLE_LOCATIONS_TRAIN)} locations...")
    start_fetch_time = time.time()
    successful_fetches = 0
    for i, (lat, lon) in enumerate(SAMPLE_LOCATIONS_TRAIN):
        location_df = fetch_nasa_power_historical_data(lat, lon, NASA_API_KEY)
        if location_df is not None:
            all_data_frames.append(location_df)
            successful_fetches += 1
        
        wait_time = 4
        print(f"Location {i+1}/{len(SAMPLE_LOCATIONS_TRAIN)} processed. Waiting {wait_time} seconds...")
        time.sleep(wait_time)
    end_fetch_time = time.time()
    print(f"\nFinished fetching data for {successful_fetches}/{len(SAMPLE_LOCATIONS_TRAIN)} locations in {end_fetch_time - start_fetch_time:.2f} seconds.")

#Combining and Preparing Data
if not all_data_frames:
    print("\n--- ERROR: No data fetched successfully. Cannot proceed. ---")
    exit()
print("\nCombining data...")
combined_data = pd.concat(all_data_frames)

combined_data.dropna(subset=TARGETS, inplace=True)
if combined_data.empty:
    print("\n--- ERROR: All fetched data had missing target values. Cannot proceed. ---")
    exit()
print(f"Total combined records after cleaning: {len(combined_data)}")
#day_of_year feature
combined_data['day_of_year'] = combined_data.index.dayofyear
print("\nFeatures used:", FEATURES)
print("Targets used:", TARGETS)


#Splitting and training data
print("\nSplitting data into training (80%) and testing (20%) sets...")
columns_to_split = FEATURES + TARGETS
train_df, test_df = train_test_split(
    combined_data[columns_to_split], test_size=0.2, random_state=42
)
X_train = train_df[FEATURES]
X_test = test_df[FEATURES]
y_solar_train = train_df[TARGETS[0]]
y_solar_test = test_df[TARGETS[0]]
y_wind_train = train_df[TARGETS[1]]
y_wind_test = test_df[TARGETS[1]]
print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")


#Model
print("\nTraining Solar Radiation model on the training set...")
start_train_time = time.time()

solar_model = RandomForestRegressor(n_estimators=100, 
                                    random_state=42,
                                    n_jobs=-1,
                                    max_depth=15, 
                                    min_samples_split=10,
                                    min_samples_leaf=5)
solar_model.fit(X_train, y_solar_train)
print("Solar Radiation model training complete.")

print("Training Wind Speed model on the training set...")
wind_model = RandomForestRegressor(n_estimators=100,
                                   random_state=42,
                                   n_jobs=-1,
                                   max_depth=15, 
                                   min_samples_split=10,
                                   min_samples_leaf=5)
wind_model.fit(X_train, y_wind_train)
end_train_time = time.time()
print("Wind Speed model training complete.")
print(f"Total model training time: {end_train_time - start_train_time:.2f} seconds.")



print("\nEvaluating models on the testing set...")
start_eval_time = time.time()
#Solar Model Evaluation
y_solar_pred = solar_model.predict(X_test)
solar_r2 = r2_score(y_solar_test, y_solar_pred)
solar_mae = mean_absolute_error(y_solar_test, y_solar_pred)
print("\n--- Solar Model Evaluation ---")
print(f"  R-squared (R²): {solar_r2:.4f}")
print(f"  Mean Absolute Error (MAE): {solar_mae:.4f} kWh/m²/day")


#Wind Model Evaluation
y_wind_pred = wind_model.predict(X_test)
wind_r2 = r2_score(y_wind_test, y_wind_pred)
wind_mae = mean_absolute_error(y_wind_test, y_wind_pred)
print("\n--- Wind Speed Model Evaluation ---")
print(f"  R-squared (R²): {wind_r2:.4f}")
print(f"  Mean Absolute Error (MAE): {wind_mae:.4f} m/s")

end_eval_time = time.time()
print(f"Total model evaluation time: {end_eval_time - start_eval_time:.2f} seconds.")




#Combining Models
combined_model_to_save = {
    'solar_model': solar_model,
    'wind_model': wind_model,
    'features': FEATURES,
    'evaluation': {
        'solar_r2': solar_r2, 'solar_mae': solar_mae,
        'wind_r2': wind_r2, 'wind_mae': wind_mae
    }
}

#Saving Model
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"\nSaving combined model to {MODEL_PATH}...")
try:
    joblib.dump(combined_model_to_save, MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")
except Exception as e:
    print(f"Error saving model: {e}")

print("\nModel training and evaluation script finished.")

