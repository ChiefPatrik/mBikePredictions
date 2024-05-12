import os
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')

for i in range(1, 30):
    # Define the current dataset file
    station_data_filename = f'station{i}_data.csv'
    station_data = pd.read_csv(os.path.join(data_dir, 'merged', station_data_filename))

    # Convert 'datetime' column to datetime
    station_data['datetime'] = pd.to_datetime(station_data['datetime'])

    # Sort the data by 'datetime' column
    station_data = station_data.sort_values('datetime')

    # Calculate the index at which to split the data
    split_index = int(len(station_data) * 0.9)

    # Split the data into training and testing sets
    train_data = station_data.iloc[:split_index]
    test_data = station_data.iloc[split_index:]

    # Define the training and testing files
    train_data_file = f'station{i}_train.csv'
    test_data_file = f'station{i}_test.csv'

    # Save the training and testing data to CSV files
    train_data.to_csv(os.path.join(data_dir, 'merged', train_data_file), index=False)
    test_data.to_csv(os.path.join(data_dir, 'merged', test_data_file), index=False)

print("Data split and saved successfully for all stations!")