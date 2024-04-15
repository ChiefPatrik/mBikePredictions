import pandas as pd
import numpy as np
import glob
import joblib
import re
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense


current_dir = os.path.dirname(os.path.abspath(__file__))
processed_data_dir = os.path.join(current_dir, '..', '..', 'data', 'processed')
models_dir = os.path.join(current_dir, '..', '..', 'models')
reports_dir = os.path.join(current_dir, '..', '..', 'reports')
window_size = 10 

# ====================
# IZRISOVANJE GRAFOV
# ====================

def plot_time_series(bike_data, selected_features):
    for feature in selected_features:
        plt.figure(figsize=(15, 6))
        plt.plot(bike_data['datetime'], bike_data[feature])
        plt.title(f'{feature} correlation with time')
        plt.xlabel('Time')
        plt.ylabel(f'{feature}')
        plt.show()

def plot_learning_history(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_time_series_predictions(bike_data, predictions, mae, mse, evs):
    plt.figure(figsize=(10, 5))
    plt.plot(bike_data['datetime'], bike_data['available_bike_stands'], label='Resnična vrednost')
    plt.plot(bike_data['datetime'][-len(predictions):], predictions, label='Napoved')
    plt.title(f'MAE: {str(round(mae, 2))}, MSE: {str(round(mse, 2))}, EVS: {str(round(evs, 2))}')
    plt.suptitle
    plt.xlabel('Time')
    plt.ylabel('available_bike_stands')
    plt.legend()
    plt.show()

def plot_last_n_values(bike_data, predictions, n, mae, mse, evs):
    plt.figure(figsize=(10, 5))
    plt.plot(bike_data['datetime'][-n:], bike_data['available_bike_stands'][-n:], label='Resnična vrednost')
    plt.plot(bike_data['datetime'][-n:], predictions[-n:], label='Napoved')
    plt.title(f'MAE: {str(round(mae, 2))}, MSE: {str(round(mse, 2))}, EVS: {str(round(evs, 2))}')
    plt.xlabel('Time')
    plt.ylabel('available_bike_stands')
    plt.legend()
    plt.show()


# ====================
# OBDELAVA PODATKOV
# ====================

def fill_missing_values(bike_data):
    print("Features with missing values BEFORE:\n", bike_data.isnull().any(), "\n")
    
    # Split the dataset into two parts: one with missing values and one without
    data_with_missing = bike_data[bike_data.isnull().any(axis=1)]
    data_without_missing = bike_data.dropna()
    columns_with_missing = bike_data.columns[bike_data.isnull().any()].tolist()
    columns_without_missing = bike_data.columns[~bike_data.isnull().any()].tolist()
    columns_without_missing = [col for col in columns_without_missing if col != 'date']

    for column in columns_with_missing:
        X_train = data_without_missing[columns_without_missing]
        y_train = data_without_missing[column]

        # RandomForestRegressor had best results in the past
        model = RandomForestRegressor(random_state=1234)
        model.fit(X_train, y_train)

        # Use the trained model to predict missing values
        X_missing = data_with_missing[columns_without_missing]
        predicted_values = model.predict(X_missing)

        # Fill in missing values in the original dataset
        bike_data.loc[bike_data.index.isin(data_with_missing.index), column] = predicted_values

    print("Features with missing values AFTER:\n", bike_data.isnull().any(), "\n")
    return bike_data

# Function for creating parts / "choppings" of time series
def create_time_series_data(time_series, window_size):
    X, y = [], []
    for i in range(len(time_series) - window_size):
        window = time_series[i:(i + window_size), :]  # V okno vključimo vse značilnice
        target = time_series[i + window_size, :]      # Target vključuje vse značilnice za naslednji časovni korak
        X.append(window)
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    return X, y[:, 0]

# Function for preparing train and test data for multivariate time series
def prepare_data_for_multivariate_time_series(bike_data, selected_features):
    df_selected_features = bike_data[selected_features].values
      
    total_size = len(df_selected_features)          # Total size of dataframe
    test_size = (total_size // 3) + window_size     # 33% of total_size + window_size
    train_size = total_size - test_size

    train_data, test_data = df_selected_features[:train_size], df_selected_features[train_size:]

    # print('Train data length:', len(train_data))  
    # print('Test data length:', len(test_data))    

    # Split dataset in 2 parts - with and without target feature
    available_bike_stands_train_data = train_data[:,0]
    available_bike_stands_test_data = test_data[:,0]

    other_train_data = train_data[:,1:]
    other_test_data = test_data[:,1:]

    # Normalize dataset
    available_bike_stands_scaler = MinMaxScaler()
    scaled_ABS_train_data = available_bike_stands_scaler.fit_transform(available_bike_stands_train_data.reshape(-1, 1))
    scaled_ABS_test_data = available_bike_stands_scaler.transform(available_bike_stands_test_data.reshape(-1, 1))

    features_scaler = MinMaxScaler()
    scaled_other_train_data = features_scaler.fit_transform(other_train_data)
    scaled_other_test_data = features_scaler.fit_transform(other_test_data)

    # Combine all traning data
    train_data = np.column_stack([
        scaled_ABS_train_data,
        scaled_other_train_data
    ])

    # Combine all test data
    test_data = np.column_stack([
        scaled_ABS_test_data,
        scaled_other_test_data
    ])

    # Create time series parts of size 'window_size'
    X_train, y_train = create_time_series_data(train_data, window_size)
    X_test, y_test = create_time_series_data(test_data, window_size)

    # print('Train data shape: X_train:', X_train.shape, ', y_train:', y_train.shape)
    # print('Test data shape: X_test:', X_test.shape, ', y_test:', y_test.shape)

    # Transpose X_train and X_test
    X_train = np.transpose(X_train, (0, 2, 1))  # Transpose axes 1 and 2
    X_test = np.transpose(X_test, (0, 2, 1))    # Transpose axes 1 and 2

    # print('Reshaped X_train:', X_train.shape)
    # print('Reshaped X_test:', X_test.shape)

    return X_train, y_train, X_test, y_test, available_bike_stands_scaler, features_scaler


# ========================================
# IZGRADNJA IN UČENJE NAPOVEDNEGA MODELA
# ========================================

def build_model(inputShape):
    model = Sequential()

    model.add(SimpleRNN(64, activation='relu', input_shape=inputShape, return_sequences=True))
    model.add(SimpleRNN(64, activation='relu'))

    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_model(available_bike_stands_scaler, model, X_test, y_test):
    # Predicting on test data
    y_pred = model.predict(X_test)

    # Inverse transformation of predicted and actual values
    y_pred_inv = available_bike_stands_scaler.inverse_transform(y_pred)
    y_test_inv = available_bike_stands_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculating metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    ev = explained_variance_score(y_test_inv, y_pred_inv)
    return mse, mae, ev, y_pred_inv

def save_data(station_number, model, available_bike_stands_scaler, features_scaler, history, mse, mae, ev):
    dir_name = f'station{station_number}'
    model_name = f'station{station_number}_model.h5'
    train_metrics_name = f'station{station_number}_train_metrics.txt'
    metrics_name = f'station{station_number}_metrics.txt'

    # Save model
    model.save(os.path.join(models_dir, dir_name, model_name))
    print("Model saved!")

    # Save scalers
    joblib.dump(available_bike_stands_scaler, '../../models/ABS_scaler.pkl')
    joblib.dump(features_scaler, '../../models/features_scaler.pkl')

    # Save train metrics    
    with open(os.path.join(reports_dir, train_metrics_name), "a") as file:
        file.write(f"Loss: {history.history['loss']} \nVal_loss: {history.history['val_loss']}\n\n")
    print(f"Train metrics for model {station_number} saved!")

    # Save test metrics
    with open(os.path.join(reports_dir, metrics_name), "a") as file:
        file.write(f"MSE: {mse}\nMAE: {mae}\nEV: {ev} \n\n")
    print(f"Test metrics {station_number} saved!")


def main():
    # Get a list of all CSV files in the directory
    csv_files = glob.glob(os.path.join(processed_data_dir, 'station*_data.csv'))

    # Select features for multivariate time series
    selected_features = [
        'available_bike_stands',
        'temperature',
        'relative_humidity',
        'dew_point',
        'apparent_temperature',
        'precipitation_probability',
        'rain',
        'surface_pressure',
        'bike_stands'
    ]
    
    for file in csv_files:
        bike_data = pd.read_csv(file)
        station_number = int(re.search('station(\d+)_data.csv', file).group(1))

        bike_data['datetime'] = pd.to_datetime(bike_data['datetime'])
        bike_data = bike_data.sort_values(by='datetime')
        bike_data = bike_data[
            ['datetime'] +
            ['available_bike_stands',
            'temperature',
            'relative_humidity',
            'dew_point',
            'apparent_temperature',
            'precipitation_probability',
            'rain',
            'surface_pressure',
            'bike_stands']
        ]

        if bike_data.isnull().any().any():
            bike_data = fill_missing_values(bike_data)

        X_train, y_train, X_test, y_test, available_bike_stands_scaler, features_scaler = prepare_data_for_multivariate_time_series(bike_data, selected_features)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        print("Input shape: ", input_shape)
        rnn_model = build_model(input_shape)
        rnn_history = rnn_model.fit(X_train, y_train, epochs=30, validation_split=0.2)
        mse, mae, ev, predictions_inv = evaluate_model(available_bike_stands_scaler, rnn_model, X_test, y_test)

        # plot_learning_history(rnn_history)
        # plot_time_series_predictions(bike_data, predictions_inv, mse, mae, ev)
        # plot_last_n_values(bike_data, predictions_inv, 10, mse, mae, ev)

        save_data(station_number, rnn_model, available_bike_stands_scaler, features_scaler, rnn_history, mse, mae, ev)


if __name__ == "__main__":
    main()