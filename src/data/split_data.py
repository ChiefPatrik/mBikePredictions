import os
import pandas as pd

# Your existing code
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, '..', '..', 'data')
current_dataset = pd.read_csv(os.path.join(data_dir, 'merged', 'current_data.csv'))

# Convert 'datetime' column to datetime
current_dataset['datetime'] = pd.to_datetime(current_dataset['datetime'])

# Sort the data by 'datetime' column
current_dataset = current_dataset.sort_values('datetime')

# Calculate the index at which to split the data
split_index = int(len(current_dataset) * 0.9)

# Split the data into training and testing sets
train_data = current_dataset.iloc[:split_index]
test_data = current_dataset.iloc[split_index:]

# Save the training and testing data to CSV files
train_data.to_csv(os.path.join(data_dir, 'merged', 'train_data.csv'), index=False)
test_data.to_csv(os.path.join(data_dir, 'merged', 'test_data.csv'), index=False)

print("Data split and saved successfully!")