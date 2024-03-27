import json
import csv
import os

def read_json(filename):
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", filename))
    with open(filepath, mode='r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_station_data(station, filename):
    fieldnames = ['number', 'contract_name', 'name', 'address', 'lat', 'lng', 'banking', 'bonus', 'bike_stands', 'available_bike_stands', 'available_bikes', 'status', 'last_update']
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", filename))
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        #writer.writeheader()  # Header row with fieldnames
        writer.writerow({
            'number': station['number'],
            'contract_name': station['contract_name'],
            'name': station['name'],
            'address': station['address'],
            'lat': station['position']['lat'],
            'lng': station['position']['lng'],
            'banking': station['banking'],
            'bonus': station['bonus'],
            'bike_stands': station['bike_stands'],
            'available_bike_stands': station['available_bike_stands'],
            'available_bikes': station['available_bikes'],
            'status': station['status'],
            'last_update': station['last_update']
        })

def main():
    json_filename = "raw_mbike_dataset.json"
    data = read_json(json_filename)

    for station in data:
        address = station['address']
        filename = f"{address.replace(' ', '_').replace('.', '')}_data.csv"
        save_station_data(station, filename)

    print("Data processed and saved to CSV files successfully!")

if __name__ == "__main__":
    main()