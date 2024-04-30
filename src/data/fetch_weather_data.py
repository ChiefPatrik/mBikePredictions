import requests
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
raw_data_path = os.path.join(data_dir, 'raw')


def fetch_weather_data(station_data): 
    # Extract latitude and longitude for each station
    coordinates_lat = ""
    coordinates_lng = ""
    for station in station_data:
        coordinates_lat += str(station['position']['lat']) + ","
        coordinates_lng += str(station['position']['lng']) + ","
    # Remove the last comma
    coordinates_lat = coordinates_lat.rstrip(',')
    coordinates_lng = coordinates_lng.rstrip(',')

    url = f"https://api.open-meteo.com/v1/forecast?latitude={coordinates_lat}&longitude={coordinates_lng}&hourly=temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,precipitation_probability,rain,surface_pressure&forecast_days=1&timezone=auto"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        weather_data = response.json()
        return weather_data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None
    
def save_to_json(data, filename):
    filepath = os.path.join(raw_data_path, filename)
    with open(filepath, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)  

def read_json(path, filename):
    filepath = os.path.join(path, filename)
    with open(filepath, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def main():
    station_json_filename = "raw_station_data.json"
    station_data = read_json(raw_data_path, station_json_filename)
    
    if station_data:
        weather_data = fetch_weather_data(station_data)
        if weather_data:
            save_to_json(weather_data, "raw_weather_data.json")
            print("Data fetched and saved successfully!")
        else:
            print("Failed to fetch weather data.")
    else:
        print("Failed to read station data from file.")


if __name__ == "__main__":
    main()