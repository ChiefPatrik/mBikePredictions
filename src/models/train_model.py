import re
import os
import glob
import onnx
import joblib
import mlflow
import dagshub
import tf2onnx
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from mlflow import MlflowClient
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from mlflow.models import infer_signature

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
# Function for filling missing values in the dataset
def fill_missing_values(bike_data):
    print("Features with missing values BEFORE:\n", bike_data.isnull().any(), "\n")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor())
    ])

    # Split the dataset into two parts: one with missing values and one without
    data_with_missing = bike_data[bike_data.isnull().any(axis=1)]
    data_without_missing = bike_data.dropna()
    columns_with_missing = bike_data.columns[bike_data.isnull().any()].tolist()
    columns_without_missing = bike_data.columns[~bike_data.isnull().any()].tolist()
    columns_without_missing = [col for col in columns_without_missing if col != 'date']

    for column in columns_with_missing:
        X_train = data_without_missing[columns_without_missing]
        y_train = data_without_missing[column]

        # model = RandomForestRegressor(random_state=1234)
        # model.fit(X_train, y_train)
        pipeline.fit(X_train, y_train)

        # Use the trained model to predict missing values
        X_missing = data_with_missing[columns_without_missing]
        #predicted_values = model.predict(X_missing)
        predicted_values = pipeline.predict(X_missing)

        # Fill in missing values in the original dataset
        bike_data.loc[bike_data.index.isin(data_with_missing.index), column] = predicted_values

    print("Features with missing values AFTER:\n", bike_data.isnull().any(), "\n")
    return bike_data, pipeline

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
def prepare_data_for_multivariate_time_series(train_data, selected_features):
    # Convert pandas dataframe to numpy array
    train_data = train_data[selected_features].values
    train_size = len(train_data) - (len(train_data) // 5)
    train, test = train_data[0:train_size], train_data[train_size:]

    # Split dataset in 2 parts - with and without target feature
    available_bike_stands_train_data = train[:,0]
    other_train_data = train[:,1:]

    available_bike_stands_test_data = test[:,0]
    other_test_data = test[:,1:]

    # Normalize dataset
    available_bike_stands_scaler = MinMaxScaler()
    scaled_ABS_train_data = available_bike_stands_scaler.fit_transform(available_bike_stands_train_data.reshape(-1, 1))

    features_scaler = MinMaxScaler()
    scaled_other_train_data = features_scaler.fit_transform(other_train_data)

    scaled_ABS_test_data = available_bike_stands_scaler.transform(available_bike_stands_test_data.reshape(-1, 1))
    scaled_other_test_data = features_scaler.transform(other_test_data)

    # Combine all training data
    train_data = np.column_stack([
        scaled_ABS_train_data,
        scaled_other_train_data
    ])
    test_data = np.column_stack([
        scaled_ABS_test_data,
        scaled_other_test_data
    ])

    # Create time series parts of size 'window_size'
    X_train, y_train = create_time_series_data(train_data, window_size)
    X_test, y_test = create_time_series_data(test_data, window_size)

    # Transpose X_train and X_test
    X_train = np.transpose(X_train, (0, 2, 1))  # Transpose axes 1 and 2
    X_test = np.transpose(X_test, (0, 2, 1))  # Transpose axes 1 and 2

    return X_train, y_train, X_test, y_test, available_bike_stands_scaler, features_scaler

# Function for processing 'datetime', filling missing values and preparing data for multivariate time series
def process_data(train_data, selected_features):
    # Convert 'datetime' column to datetime and sort the data
    train_data['datetime'] = pd.to_datetime(train_data['datetime'])
    train_data = train_data.sort_values(by='datetime')
    train_data = train_data[['datetime'] + selected_features]

    # Fill missing values
    if train_data.isnull().any().any():
        train_data, pipeline = fill_missing_values(train_data)

    # Split the dataset into training and testing parts
    X_train, y_train, X_test, y_test, available_bike_stands_scaler, features_scaler = prepare_data_for_multivariate_time_series(train_data, selected_features) 	
    input_shape = (X_train.shape[1], X_train.shape[2])

    return X_train, y_train, X_test, y_test, available_bike_stands_scaler, features_scaler, input_shape


# ========================
# SHRANJEVANJE VREDNOSTI
# ========================
# Function for saving model, scalers and train metrics locally
def save_data(station_number, model, available_bike_stands_scaler, features_scaler, learnHistory): 
    dir_name = f'station{station_number}'
    model_name = f'station{station_number}_model.h5'
    train_metrics_name = f'station{station_number}_train_metrics.txt'

    # Save model
    model.save(os.path.join(models_dir, dir_name, model_name))
    print("Model saved!")

    # Save scalers
    joblib.dump(available_bike_stands_scaler, os.path.join(models_dir, dir_name, 'ABS_scaler.pkl'))
    joblib.dump(features_scaler, os.path.join(models_dir, dir_name, 'features_scaler.pkl'))

    # Save train metrics
    dir_path = os.path.join(reports_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    with open(os.path.join(dir_path, train_metrics_name), "a") as file:
        file.write(f"Loss: {learnHistory.history['loss']} \nVal_loss: {learnHistory.history['val_loss']}\n\n")
    print(f"Train metrics for model {station_number} saved!")

# Function for saving scalers to MLflow
def mlflow_save_scaler(client, scaler_name, scaler, station_number, stage_param="Staging"):
    metadata = {
        "station_number": station_number,
        "scaler_name": scaler_name,
        "expected_features": scaler.n_features_in_,
        "feature_range": scaler.feature_range,
    }

    scaler = mlflow.sklearn.log_model(
        sk_model=scaler,
        artifact_path=f"models/{station_number}/{scaler_name}",
        registered_model_name=f"{scaler_name}={station_number}",
        metadata=metadata,
    )

    scaler_version = client.create_model_version(
        name=f"{scaler_name}={station_number}",
        source=scaler.model_uri,
        run_id=scaler.run_id
    )

    client.transition_model_version_stage(
        name=f"{scaler_name}={station_number}",
        version=scaler_version.version,
        stage=stage_param,
    )


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

def get_metric_from_run(client, run_id, metric_name):
    metric = client.get_metric_history(run_id, metric_name)
    return metric[-1].value if metric else None


def main():
    # Get all train files
    train_files = glob.glob(os.path.join(merged_data_dir,'station*_train.csv'))

    train_files.sort()
    train_files = train_files[:5]

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

    client = MlflowClient()

    for train_file in train_files:
        station_number = int(re.search('station(\d+)_train.csv', train_file).group(1))
        experiment_name = f"station{station_number}_train_exp"
        model_name = f"station{station_number}_model"
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"station{station_number}_train_run"):
            mlflow.autolog()
            run_id = mlflow.active_run().info.run_id    # For versioning mlflow models

            train_data = pd.read_csv(train_file)
            print(f"\nTraining model for station {station_number} ...")

            X_train, y_train, X_test, y_test, available_bike_stands_scaler, features_scaler, input_shape = process_data(train_data, selected_features)

            # Build and train the model
            rnn_model = build_model(input_shape)
            rnn_history = rnn_model.fit(X_train, y_train, epochs=30, validation_split=0.2)
            predictions = rnn_model.predict(X_test)

            # Log train metrics to MLflow
            for i in range(len(rnn_history.history['loss'])):
                mlflow.log_metric("loss", rnn_history.history['loss'][i], step=i)
                mlflow.log_metric("val_loss", rnn_history.history['val_loss'][i], step=i)
    
            # Convert the model to ONNX format
            input_signature = [
                tf.TensorSpec(shape=(None, X_train.shape[1], X_train.shape[2]), dtype=tf.double, name="input")
            ]
            onnx_model, _ = tf2onnx.convert.from_keras(model=rnn_model, input_signature=input_signature, opset=13)

            # Log the model to MLflow
            registered_model = mlflow.onnx.log_model(onnx_model=onnx_model, 
                                  artifact_path=f"models/{station_number}/model", 
                                  signature=infer_signature(X_test, predictions), 
                                  registered_model_name=model_name)
            model_version = client.create_model_version(name=model_name, source=registered_model.model_uri, run_id=run_id)    

            # Get the last production model and compare it to current
            try:
                model_versions = client.get_latest_versions(model_name, stages=["Production"])
                if model_versions:
                    last_production_version = model_versions[0]
                    last_production_run_id = last_production_version.run_id

                    # Retrieve metrics of the last production model
                    production_val_loss = get_metric_from_run(client, last_production_run_id, "val_loss")

                    # Retrieve metrics of the current model
                    current_val_loss = get_metric_from_run(client, run_id, "val_loss")

                    print(f"Current model val_loss: {current_val_loss}")
                    print(f"Production model val_loss: {production_val_loss}")

                    if current_val_loss < production_val_loss:
                        print("Current model is better. Transitioning it to production.")
                        client.transition_model_version_stage(
                            name=model_name,
                            version=model_version.version,
                            stage="Production"
                        )
                        mlflow_save_scaler(client, "ABS_scaler", available_bike_stands_scaler, station_number, "Production")
                        mlflow_save_scaler(client, "features_scaler", features_scaler, station_number, "Production")
                    else:
                        print("Current model is not better than the production model. Transitioning to staging.")
                        client.transition_model_version_stage(
                            name=model_name,
                            version=model_version.version,
                            stage="Staging"
                        )
                        mlflow_save_scaler(client, "ABS_scaler", available_bike_stands_scaler, station_number, "Staging")
                        mlflow_save_scaler(client, "features_scaler", features_scaler, station_number, "Staging")
                else:
                    print("No production model found. Transitioning current model to production.")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=model_version.version,
                        stage="Production"
                    )
                    mlflow_save_scaler(client, "ABS_scaler", available_bike_stands_scaler, station_number, "Production")
                    mlflow_save_scaler(client, "features_scaler", features_scaler, station_number, "Production")
            except Exception as e:
                print(f"Error while comparing models: {e}")
                print("Transitioning current model to staging.")
                client.transition_model_version_stage(
                    name=model_name,
                    version=model_version.version,
                    stage="Staging"
                )
                mlflow_save_scaler(client, "ABS_scaler", available_bike_stands_scaler, station_number, "Staging")
                mlflow_save_scaler(client, "features_scaler", features_scaler, station_number, "Staging")
        
        mlflow.end_run()

        # plot_learning_history(rnn_history)
        # plot_time_series_predictions(bike_data, predictions_inv, mse, mae, ev)
        # plot_last_n_values(bike_data, predictions_inv, 10, mse, mae, ev)

        save_data(station_number, rnn_model, available_bike_stands_scaler, features_scaler, rnn_history)  


if __name__ == "__main__":
    main()