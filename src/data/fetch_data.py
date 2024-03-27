import requests
import json
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

def save_to_json(data, filename):
    filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", filename))
    with open(filepath, mode='w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def main():
    contract = "maribor" 
    api_key = "5e150537116dbc1786ce5bec6975a8603286526b"  
    station_data = fetch_station_data(contract, api_key)
    
    if station_data:
        save_to_json(station_data, "raw_mbike_dataset.json")
        print("Data fetched and saved successfully!")
    else:
        print("Failed to fetch station data.")

if __name__ == "__main__":
    main()