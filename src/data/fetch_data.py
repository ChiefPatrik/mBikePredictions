import requests
import pandas as pd
import csv
import os


def fetch_station_data(contract, api_key):
    url = f"https://api.jcdecaux.com/vls/v1/stations?contract={contract}&apiKey={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise exception for bad status codes
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None


def save_to_csv(data, filename):
    fieldnames = ['number', 'contract_name', 'name', 'address', 'lat', 'lng', 'banking', 'bonus', 'bike_stands', 'available_bike_stands', 'available_bikes', 'status', 'last_update']
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", filename))
    with open(filepath, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Header row with fieldnames
        for station in data:
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
    contract = "maribor" 
    api_key = "5e150537116dbc1786ce5bec6975a8603286526b"  
    station_data = fetch_station_data(contract, api_key)
    
    if station_data:
        save_to_csv(station_data, "raw_mbike_dataset.csv")
        print("Data saved successfully!")
    else:
        print("Failed to fetch station data.S")

if __name__ == "__main__":
    main()