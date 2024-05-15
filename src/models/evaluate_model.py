import os
import mlflow
import glob
import mlflow
import dagshub
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score


current_dir = os.path.dirname(os.path.abspath(__file__))
merged_data_dir = os.path.join(current_dir, '..', '..', 'data', 'merged')
models_dir = os.path.join(current_dir, '..', '..', 'models')
reports_dir = os.path.join(current_dir, '..', '..', 'reports')
window_size = 30

# Setup MLflow and Dagshub
load_dotenv()
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
dagshub.auth.add_app_token(token=mlflow_password)
dagshub.init(repo_owner=mlflow_username, repo_name='mBikePredictions', mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)


# ====================
# OBDELAVA PODATKOV
# ====================

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

# Function for preparing test data for multivariate time series
def prepare_data_for_multivariate_time_series(test_data, selected_features, abs_scaler):
    # Convert pandas dataframe to numpy array
    test_data = test_data[selected_features].values

    # Split dataset in 2 parts - with and without target feature
    available_bike_stands_test_data = test_data[:,0]
    other_test_data = test_data[:,1:]

    # Normalize dataset
    scaled_ABS_test_data = abs_scaler.transform(available_bike_stands_test_data.reshape(-1, 1))

    features_scaler = MinMaxScaler()
    scaled_other_test_data = features_scaler.fit_transform(other_test_data)

    # Combine all test data
    test_data = np.column_stack([
        scaled_ABS_test_data,
        scaled_other_test_data
    ])

    # Create time series parts of size 'window_size'
    X_test, y_test = create_time_series_data(test_data, window_size)

    # Transpose X_test
    X_test = np.transpose(X_test, (0, 2, 1))    # Transpose axes 1 and 2

    return X_test, y_test


# ========================================
# EVALUACIJA NAPOVEDNEGA MODELA
# ========================================  

# Function for evaluating model with mse, mae and ev metrics
def evaluate_model(station_number, available_bike_stands_scaler, model, X_test, y_test):
    mlflow.set_experiment(f"station{station_number}_test_exp")
    with mlflow.start_run(run_name=f"station{station_number}_test_run"):
        mlflow.autolog()
        # Predicting on test data
        y_pred = model.predict(X_test)

        # Inverse transformation of predicted and actual values
        y_pred_inv = available_bike_stands_scaler.inverse_transform(y_pred)
        y_test_inv = available_bike_stands_scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculating metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        ev = explained_variance_score(y_test_inv, y_pred_inv)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("EVS", ev)

    mlflow.end_run()
    return mse, mae, ev

def save_data(station_number, mse, mae, ev):
    dir_name = f'station{station_number}'
    metrics_name = f'station{station_number}_metrics.txt'

    # Save test metrics
    with open(os.path.join(reports_dir, dir_name, metrics_name), "a") as file:
        file.write(f"MSE: {mse}\nMAE: {mae}\nEV: {ev} \n\n")
    print(f"Test metrics for station{station_number} saved!")


def main():
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

    for model_dir in glob.glob(os.path.join(models_dir, '*')):
        # Extract station number from directory name
        station_number = os.path.basename(model_dir).replace('station', '')

        # Load model and scaler
        model = load_model(os.path.join(model_dir, f'station{station_number}_model.h5'))
        scaler = joblib.load(os.path.join(model_dir, 'ABS_scaler.pkl'))

        # Load corresponding CSV file
        csv_path = os.path.join(merged_data_dir, f'station{station_number}_test.csv')
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)

        data['datetime'] = pd.to_datetime(data['datetime'])
        data = data.sort_values(by='datetime')

        X_test, y_test = prepare_data_for_multivariate_time_series(data, selected_features, scaler)

        mse, mae, ev = evaluate_model(station_number, scaler, model, X_test, y_test)

        save_data(station_number, mse, mae, ev)

if __name__ == "__main__":
    main()