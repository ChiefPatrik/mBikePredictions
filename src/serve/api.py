import os
import ast
import onnx
import joblib
import dagshub
import uvicorn
import requests
import numpy as np
import pandas as pd
import onnxruntime as onnx_rt
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel
from pymongo import MongoClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
#from tensorflow.keras.models import load_model
import mlflow
import mlflow.keras
import mlflow.sklearn
from mlflow.tracking import MlflowClient
#from src.models.predict_model import predict


current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
merged_data_dir = os.path.join(current_dir, '..', '..', 'data', 'merged')

window_size = 30
time_interval = 7


# Setup Dagshub, MLflow and MongoDB
load_dotenv()
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mongo_username = os.getenv("MONGO_USERNAME")
mongo_password = os.getenv("MONGO_PASSWORD")

dagshub.auth.add_app_token(token=mlflow_password)
dagshub.init(repo_owner=mlflow_username, repo_name='mBikePredictions', mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)

db_name = "mBikeInputData"
connection_string = f"mongodb+srv://{mongo_username}:{mongo_password}@cluster0.v3fli1i.mongodb.net/"
client = MongoClient(connection_string)

# Function for downloading models and scalers from MLflow
def download_models():
    client = MlflowClient()
    print("Presaving models and scalers ...")
    for i in range(10,15):
        if i == 11:
            continue
        model_name = f"station{i}_model"
        ABS_scaler_name = f"ABS_scaler={i}"
        features_scaler_name = f"features_scaler={i}"

        model = mlflow.onnx.load_model(client.get_latest_versions(name=model_name, stages=["Production"])[0].source)
        ABS_scaler = mlflow.sklearn.load_model( client.get_latest_versions(name=ABS_scaler_name, stages=["Production"])[0].source)
        features_scaler = mlflow.sklearn.load_model( client.get_latest_versions(name=features_scaler_name, stages=["Production"])[0].source)

        onnx.save_model(model, os.path.join(models_dir, f'station{i}', f"station{i}_model.onnx"))
        joblib.dump(ABS_scaler, os.path.join(models_dir, f'station{i}', 'ABS_scaler.pkl'))
        joblib.dump(features_scaler, os.path.join(models_dir, f'station{i}', 'features_scaler.pkl'))

        print("Downloaded model and scalers for station", i)
    
    print("All models downloaded and saved locally!")

# Function for saving input data to MongoDB    
def save_to_mongodb(first_prediction_datetime, predicted_values): 
    db = client.get_database(db_name)  
    collection = db["inputDatasets"]
    
    # Convert the DataFrame to a list of dictionaries for insertion into MongoDB
    data_dict = {
        "datetime": first_prediction_datetime,
        "predicted_values": predicted_values
    }
    
    collection.insert_one(data_dict)
    print("Input data saved to MongoDB!")

# Function for fetching fresh weather data from Open-Meteo API
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

# Function for constructing a new dataset on top of old one with additional fresh weather data    
def construct_dataset(station_data, weather_data):
    last_rows = station_data.tail(window_size)  # Select the last 'window_size' rows from station_data

    last_datetime = pd.to_datetime(last_rows['datetime'].values[-1])            # Get the last datetime value from station_data

    weather_datetimes = pd.to_datetime(weather_data['hourly']['time'])          # Convert the datetime values in weather_data to datetime objects

    next_datetime_index = weather_datetimes.searchsorted(last_datetime) + 1     # Find the index of the next datetime value in weather_data

    if next_datetime_index == len(weather_datetimes):       # If the next datetime value is not in weather_data, raise an error
        raise ValueError("The next datetime value is not in weather_data")

    # Select the next 'time_interval' values for each attribute from next_datetime_index in weather_data
    next_values = {attr: weather_data['hourly'][attr][next_datetime_index:next_datetime_index + time_interval] for attr in [
        'time', 'temperature_2m', 'relative_humidity_2m', 'dew_point_2m', 'apparent_temperature', 
        'precipitation_probability', 'rain', 'surface_pressure']}

    new_rows = pd.DataFrame(next_values)    # ... and create a new DataFrame with those values

    # Rename the columns to match the columns in station_data
    new_rows.columns = ['datetime', 'temperature', 'relative_humidity', 'dew_point', 'apparent_temperature', 
                        'precipitation_probability', 'rain', 'surface_pressure']

    # Copy the columns from last_rows that are not in new_rows
    for col in ['number', 'name', 'address', 'coordinates', 'bike_stands']:
        new_rows[col] = last_rows[col].values[0]

    # Convert 'datetime' to datetime objects
    new_rows['datetime'] = pd.to_datetime(new_rows['datetime'])
    # Format 'datetime' as 'YYYY-MM-DD HH:MM:SS'
    new_rows['datetime'] = new_rows['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    # Leave the 'available_bike_stands' column empty
    new_rows['available_bike_stands'] = np.nan      
    first_datetime = new_rows['datetime'].values[0]
    #print("First datetime: ", first_datetime)

    # Concatenate last_rows (from stationX_data.csv) and new_rows (from weather_data) to create a new dataset
    dataset = pd.concat([last_rows, new_rows], ignore_index=True)
    #print("DATASET BEFORE", dataset)

    return dataset, first_datetime


def predict_next_time_interval(dataset, model, ABS_scaler, features_scaler):
    predicted_values = []

    # Predict 'time_interval' steps ahead
    for counter in range(time_interval):
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
        complete_data_reshaped = complete_data.reshape(1, complete_data.shape[1], complete_data.shape[0])    # Transpose axes 1 and 2

        #y_pred = model.predict(complete_data_reshaped)
        input_name = model.get_inputs()[0].name
        y_pred = model.run(None, {input_name: complete_data_reshaped.astype(np.double)})[0]

        next_value = ABS_scaler.inverse_transform(y_pred)
        print("Next value: ", next_value)
        if(next_value[0][0] < 0):
            next_value[0][0] = 0

        # Add the predicted value to predicted_values
        predicted_values.append(int(next_value[0][0]))

        # Add the predicted value to input_data for next iteration of prediction
        dataset.at[window_size + counter, 'available_bike_stands'] = int(next_value[0][0])
    
    return predicted_values

def predict(station_number):
    # model = load_model(os.path.join(models_dir, f'station{station_number}', model_name))
    ABS_scaler = joblib.load(os.path.join(models_dir, f'station{station_number}', 'ABS_scaler.pkl'))
    features_scaler = joblib.load(os.path.join(models_dir, f'station{station_number}', 'features_scaler.pkl'))

    #model, ABS_scaler, features_scaler = models_scalers[station_number].values()
    model = onnx_rt.InferenceSession(os.path.join(models_dir, f'station{station_number}', f"station{station_number}_model.onnx"))

    # Load existing station data
    station_data = pd.read_csv(os.path.join(merged_data_dir, f'station{station_number}_data.csv'))
    
    # Fetch fresh weather data for given station
    first_row = station_data.iloc[1]
    location_dict = ast.literal_eval(first_row['coordinates'])
    weather_data = fetch_fresh_weather_data(location_dict['lat'], location_dict['lng'])

    dataset, first_prediction_datetime = construct_dataset(station_data, weather_data)
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

    predicted_values = predict_next_time_interval(dataset, model, ABS_scaler, features_scaler)
    save_to_mongodb(first_prediction_datetime, predicted_values)

    #print("Dataset after:\n", dataset)
    print("Predicted values: ", predicted_values)
    return predicted_values


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
    download_models()
    uvicorn.run(app, host="0.0.0.0", port=3001)