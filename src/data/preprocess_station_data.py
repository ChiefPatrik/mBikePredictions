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

def save_station_data(station, filename):
    fieldnames = ['number',
                  'name',
                  'address',
                  'coordinates', 
                  'bike_stands', 
                  'available_bike_stands'
                  ]
    filepath = os.path.join(processed_data_path, filename)
    with open(filepath, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        #writer.writeheader()    # Header row with fieldnames        
        writer.writerow({
            'number': station['number'],
            'name': station['name'],
            'address': station['address'],
            'coordinates': station['position'],
            'bike_stands': station['bike_stands'],
            'available_bike_stands': station['available_bike_stands']
        })

def process_data(station_data):
    for index, station in enumerate(station_data):
        filename = f"station{station['number']}_data.csv"
        save_station_data(station, filename)
    print("Station data preprocessed and saved to CSV files successfully!")


def main():
    station_json_filename = "raw_station_data.json"
    station_data = read_json(raw_data_path, station_json_filename)
    process_data(station_data)

if __name__ == "__main__":
    main()