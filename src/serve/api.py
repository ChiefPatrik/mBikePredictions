from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

#from src.models.predict_model import predict_model
import pandas as pd
import os
import numpy as np
import joblib
import requests
import ast
from tensorflow.keras.models import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
processed_data_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
window_size = 20
time_interval = 7

# Create server
app = FastAPI()
# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_fresh_weather_data(coordinates_lat, coordinates_lng):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coordinates_lat}&longitude={coordinates_lng}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure&forecast_days=2&timezone=auto"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    
def predict(station_number):
    model_name = f'station{station_number}_model.h5'
    
    model = load_model(os.path.join(models_dir, f'station{station_number}', model_name))
    ABS_scaler = joblib.load(os.path.join(models_dir, f'station{station_number}', 'ABS_scaler.pkl'))
    features_scaler = joblib.load(os.path.join(models_dir, f'station{station_number}', 'features_scaler.pkl'))

    # Load existing station data
    station_data = pd.read_csv(os.path.join(processed_data_dir, f'station{station_number}_data.csv'))
    
    # Fetch fresh weather data for given station
    first_row = station_data.iloc[1]
    location_dict = ast.literal_eval(first_row['coordinates'])
    weather_data = fetch_fresh_weather_data(location_dict['lat'], location_dict['lng'])


    # Select the last 'window_size' rows from station_data
    last_rows = station_data.tail(window_size)

    # Get the last datetime value from station_data
    last_datetime = pd.to_datetime(last_rows['datetime'].values[-1])

    # Convert the datetime values in weather_data to datetime objects
    weather_datetimes = pd.to_datetime(weather_data['hourly']['time'])

    # Find the index of the next datetime value in weather_data
    next_datetime_index = weather_datetimes.searchsorted(last_datetime) + 1

    # If the next datetime value is not in weather_data, raise an error
    if next_datetime_index == len(weather_datetimes):
        raise ValueError("The next datetime value is not in weather_data")

    # Select the next 'time_interval' values from each attribute in weather_data
    next_values = {attr: weather_data['hourly'][attr][next_datetime_index:next_datetime_index + time_interval] for attr in [
        'time', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature', 
        'precipitation_probability', 'rain', 'surface_pressure']}

    # Create a new DataFrame with these values
    new_rows = pd.DataFrame(next_values)

    # Rename the columns to match the columns in station_data
    new_rows.columns = ['datetime', 'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
                        'precipitation_probability', 'rain', 'surface_pressure']

    # Copy the 'number', 'name', 'address', 'coordinates', 'bike_stands' columns from last_rows
    for col in ['number', 'name', 'address', 'coordinates', 'bike_stands']:
        new_rows[col] = last_rows[col].values[0]

    # Convert 'datetime' to datetime objects
    new_rows['datetime'] = pd.to_datetime(new_rows['datetime'])
    # Format 'datetime' as 'YYYY-MM-DD HH:MM:SS'
    new_rows['datetime'] = new_rows['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Leave the 'available_bike_stands' column empty
    new_rows['available_bike_stands'] = np.nan


    # Concatenate last_rows and new_rows
    dataset = pd.concat([last_rows, new_rows], ignore_index=True)

    print(dataset)

    # Initialize a list to store the predicted values
    selected_features = [
        'available_bike_stands',
        'temperature',
        'relative_humidity',
        'dew_point',
        'apparent_temperature',
        'precipitation_probability',
        'rain',
        'surface_pressure',
        'bike_stands'
    ]
    dataset = dataset[selected_features]
    predicted_values = []
    print("Dataset after:\n", dataset)

    # Predict 7 steps ahead
    for counter in range(7):
        input_data = dataset[counter:window_size+counter].values

        available_bike_stands_data = input_data[:, 0]
        other_data = input_data[:, 1:]

        scaled_ABS_data = ABS_scaler.transform(available_bike_stands_data.reshape(-1, 1))
        scaled_other_data = features_scaler.transform(other_data)

        complete_data = np.column_stack([
            scaled_ABS_data,
            scaled_other_data
        ])
        print("shape: ", complete_data.shape)
        #complete_data_reshaped = np.transpose(complete_data, (0, 2, 1))    # Transpose axes 1 and 2
        complete_data_reshaped = complete_data.reshape(1, complete_data.shape[1], complete_data.shape[0])    # Transpose axes 1 and 2

        next_value = model.predict(complete_data_reshaped)
        print("Next value: ", next_value)

        # Add the predicted value to predicted_values
        predicted_values.append(int(next_value[0][0]))

        # Add the predicted value to input_data
        dataset.at[window_size + counter, 'available_bike_stands'] = int(next_value[0][0])

    print("Dataset after: ", dataset)    
    print("Predicted values: ", predicted_values)
    return predicted_values


class StationData(BaseModel):
    station_number: int

class PredictionInput(BaseModel):
    data: List[StationData]

@app.post("/mbajk/predict/", response_model=dict)
async def predict_mBike(input_data: PredictionInput):
    station_number = input_data.data[0].station_number
    try:   
        result = predict(station_number)
        return {'predictions': result} 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)