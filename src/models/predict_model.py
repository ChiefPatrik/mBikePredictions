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
window_size = 5


def fetch_fresh_weather_data(coordinates_lat, coordinates_lng):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={coordinates_lat}&longitude={coordinates_lng}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure&forecast_days=1&timezone=auto"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None

def sortByDate(dataframe):
    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    dataframe = dataframe.sort_values('datetime')
    return dataframe

def main(station_number):
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
    next_datetime_index = weather_datetimes.searchsorted(last_datetime)

    # If the next datetime value is not in weather_data, raise an error
    if next_datetime_index == len(weather_datetimes):
        raise ValueError("The next datetime value is not in weather_data")


    # Select the next 7 values from each attribute in weather_data
    next_values = {attr: weather_data['hourly'][attr][next_datetime_index:next_datetime_index+7] for attr in [
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

    # Leave the 'available_bike_stands' column empty
    new_rows['available_bike_stands'] = np.nan

    # Append new_rows to last_rows
    last_rows = last_rows.append(new_rows, ignore_index=True)

    print(station_data.columns)
    print(last_rows)




    # available_bike_stands = bike_data['available_bike_stands'].values.reshape(-1, 1)
    # n_available_bike_stands = standard_scaler.transform(available_bike_stands)
    # n_available_bike_stands = np.reshape(n_available_bike_stands, (n_available_bike_stands.shape[1], 1, n_available_bike_stands.shape[0]))

    # prediction = model.predict(n_available_bike_stands)[0]

    # inversed_prediction = standard_scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
    return {"prediction": int("15")}
