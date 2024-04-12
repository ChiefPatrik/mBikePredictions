import json
import csv
import os
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')

raw_data_path = os.path.join(data_dir, 'raw')
processed_data_path = os.path.join(data_dir, 'processed')


def read_json(path, filename):
    filepath = os.path.join(path, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_index_for_current_time(time_list):
    current_time = datetime.now()
    closest_time = min(time_list, key=lambda x: abs(datetime.fromisoformat(x) - current_time))
    return time_list.index(closest_time)


def format_datetime(datetime_str):
    dt_obj = datetime.fromisoformat(datetime_str)   # Convert string to datetime object
    formatted_datetime = dt_obj.strftime('%Y-%m-%d %H:%M:%S%z')
    return formatted_datetime


def save_station_data(station, weather, filename):
    fieldnames = ['number', 
                  'datetime', 
                  'temperature', 
                  'relative_humidity', 
                  'dew_point', 
                  'apparent_temperature', 
                  'precipitation_probability', 
                  'rain', 
                  'surface_pressure',
                  'bike_stands', 
                  'available_bike_stands'
                  ]
    filepath = os.path.join(processed_data_path, filename)
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        #writer.writeheader()  # Header row with fieldnames
        writer.writerow({
            'number': station['number'],
            'datetime': format_datetime(weather['datetime']),
            'temperature': weather['temperature_2m'],
            'relative_humidity': weather['relative_humidity_2m'],
            'dew_point': weather['dew_point_2m'],
            'apparent_temperature': weather['apparent_temperature'],
            'precipitation_probability': weather['precipitation_probability'],
            'rain': weather['rain'],
            'surface_pressure': weather['surface_pressure'],
            'bike_stands': station['bike_stands'],
            'available_bike_stands': station['available_bike_stands']
        })


def main():
    station_json_filename = "raw_station_data.json"
    weather_json_filename = "raw_weather_data.json"
    station_data = read_json(raw_data_path, station_json_filename)
    weather_data = read_json(raw_data_path, weather_json_filename)

    for i, station in enumerate(station_data):
        if i < len(weather_data):
            weather_entry = weather_data[i]
            current_time_index = get_index_for_current_time(weather_entry['hourly']['time'])
            weather_attributes = {
                'datetime': weather_entry['hourly']['time'][current_time_index],
                'temperature_2m': weather_entry['hourly']['temperature_2m'][current_time_index],
                'relative_humidity_2m': weather_entry['hourly']['relative_humidity_2m'][current_time_index],
                'dew_point_2m': weather_entry['hourly']['dew_point_2m'][current_time_index],
                'apparent_temperature': weather_entry['hourly']['apparent_temperature'][current_time_index],
                'precipitation_probability': weather_entry['hourly']['precipitation_probability'][current_time_index],
                'rain': weather_entry['hourly']['rain'][current_time_index],
                'surface_pressure': weather_entry['hourly']['surface_pressure'][current_time_index]
            }
            filename = f"station{station['number']}_data.csv"
            save_station_data(station, weather_attributes, filename)
        else:
            print(f"Weather data not found for station {station['number']}")

    print("Data processed and saved to CSV files successfully!")

if __name__ == "__main__":
    main()