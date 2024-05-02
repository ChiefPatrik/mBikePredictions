import os
import csv

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')

processed_data_path = os.path.join(data_dir, 'processed')
merged_data_path = os.path.join(data_dir, 'merged')

def merge_data(station_number):
    station_data_filename = f"station{station_number}_data.csv"
    weather_data_filename = f"station{station_number}_weather_data.csv"
    
    station_data_filepath = os.path.join(processed_data_path, station_data_filename)
    weather_data_filepath = os.path.join(processed_data_path, weather_data_filename)
    
    merged_data_filepath = os.path.join(merged_data_path, f"station{station_number}_data.csv")
    
    with open(station_data_filepath, mode='r', encoding='utf-8') as station_file, \
         open(weather_data_filepath, mode='r', encoding='utf-8') as weather_file, \
         open(merged_data_filepath, mode='a', newline='', encoding='utf-8') as merged_file:
        
        station_reader = csv.DictReader(station_file)
        weather_reader = csv.DictReader(weather_file)
        
        merged_fieldnames = ['number', 
                             'datetime',
                             'name',
                             'address',
                             'coordinates', 
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
        merged_writer = csv.DictWriter(merged_file, fieldnames=merged_fieldnames)
        if not os.path.isfile(merged_file):     # Header row with fieldnames
            merged_writer.writeheader()
        
        last_station_row = None
        for station_row in station_reader:
            last_station_row = station_row
        
        last_weather_row = None
        for weather_row in weather_reader:
            last_weather_row = weather_row

        merged_row = {
            'number': last_station_row['number'],
            'datetime': last_weather_row['datetime'],
            'name': last_station_row['name'],
            'address': last_station_row['address'],
            'coordinates': last_station_row['coordinates'],
            'temperature': last_weather_row['temperature'],
            'relative_humidity': last_weather_row['relative_humidity'],
            'dew_point': last_weather_row['dew_point'],
            'apparent_temperature': last_weather_row['apparent_temperature'],
            'precipitation_probability': last_weather_row['precipitation_probability'],
            'rain': last_weather_row['rain'],
            'surface_pressure': last_weather_row['surface_pressure'],
            'bike_stands': last_station_row['bike_stands'],
            'available_bike_stands': last_station_row['available_bike_stands']
        }
        merged_writer.writerow(merged_row)


def main():
    # Create the 'merged' directory if it doesn't exist
    os.makedirs(merged_data_path, exist_ok=True)
    
    # Get the list of station numbers from processed data
    station_numbers = set()
    for filename in os.listdir(processed_data_path):
        if filename.endswith('_data.csv'):
            station_numbers.add(int(filename.split('_')[0][7:]))
    
    # Merge data for each station
    for station_number in station_numbers:
        merge_data(station_number)
    
    print("Station and weather data merged successfully!")

if __name__ == "__main__":
    main()
