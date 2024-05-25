import os
import mlflow
import glob
import mlflow
import onnx
import dagshub
import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import mlflow
# import mlflow.keras
# import mlflow.sklearn
from mlflow.tracking import MlflowClient

current_dir = os.path.dirname(os.path.abspath(__file__))
merged_data_dir = os.path.join(current_dir, '..', '..', 'data', 'merged')
models_dir = os.path.join(current_dir, '..', '..', 'models')
time_interval = 7

# Setup Dagshub, MLflow and MongoDB
load_dotenv()
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
mongo_username = os.getenv("MONGO_USERNAME")
mongo_password = os.getenv("MONGO_PASSWORD")

dagshub.auth.add_app_token(token=mlflow_password)
dagshub.init(repo_owner=mlflow_username, repo_name='mBikePredictions', mlflow=True)
mlflow.set_tracking_uri(mlflow_uri)

db_name = "mBikeInputData"
connection_string = f"mongodb+srv://{mongo_username}:{mongo_password}@cluster0.v3fli1i.mongodb.net/"
client = MongoClient(connection_string)


# Function for downloading models and scalers from MLflow
def download_models_and_scalers():
    mlflowClient = MlflowClient()
    print("Presaving models and scalers ...")
    for i in range(10,15):
        if i == 11:
            continue
        model_name = f"station{i}_model"
        ABS_scaler_name = f"ABS_scaler={i}"
        features_scaler_name = f"features_scaler={i}"

        model = mlflow.onnx.load_model(mlflowClient.get_latest_versions(name=model_name, stages=["Production"])[0].source)
        ABS_scaler = mlflow.sklearn.load_model(mlflowClient.get_latest_versions(name=ABS_scaler_name, stages=["Production"])[0].source)
        features_scaler = mlflow.sklearn.load_model(mlflowClient.get_latest_versions(name=features_scaler_name, stages=["Production"])[0].source)

        onnx.save_model(model, os.path.join(models_dir, f'station{i}', f"station{i}_model.onnx"))
        joblib.dump(ABS_scaler, os.path.join(models_dir, f'station{i}', 'ABS_scaler.pkl'))
        joblib.dump(features_scaler, os.path.join(models_dir, f'station{i}', 'features_scaler.pkl'))

        print("Downloaded model and scalers for station", i)
    
    print("All models downloaded and saved locally!")

# Function for getting documents from MongoDB    
def get_predictions_from_mongo(): 
    db = client.get_database(db_name)  
    collection = db["inputDatasets"]
    documents = collection.find()
    return list(documents)

def evaluate_predictions(df, index, station_number, available_bike_stands_scaler, prediction):
    y_pred = prediction['predicted_values']
    y_test = df['available_bike_stands'].iloc[index + 1 : index + time_interval + 1].values
    for i, value in enumerate(y_pred):
        print("prediction: ", value, " actual: ", y_test[i])
    y_pred = np.array(y_pred).reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    mlflow.set_experiment(f"station{station_number}_predictions_eval_exp")
    with mlflow.start_run(run_name=f"station{station_number}_predictions_eval_run"):
        # Inverse transformation of predicted and actual values
        y_pred_inv = available_bike_stands_scaler.inverse_transform(y_pred)
        y_test_inv = available_bike_stands_scaler.inverse_transform(y_test)

        # Calculating metrics
        mse = mean_squared_error(y_test_inv, y_pred_inv)
        mae = mean_absolute_error(y_test_inv, y_pred_inv)
        ev = explained_variance_score(y_test_inv, y_pred_inv)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("EVS", ev)
    mlflow.end_run()

    return mse, mae, ev


def main():
    download_models_and_scalers()
    predictions = get_predictions_from_mongo()
    # predictions = [
    #     {
    #         "datetime": "2024-05-02 06:00:00",
    #         "station_number": 10,
    #         "predicted_values": [1, 2, 3, 4, 5, 6, 7]
    #     },
    #     {
    #         "datetime": "2024-04-15 11:00:00",
    #         "station_number": 11,
    #         "predicted_values": [8, 9, 10, 11, 12, 13, 14]
    #     },
    #     {
    #         "datetime": "2024-05-21 21:00:00",
    #         "station_number": 12,
    #         "predicted_values": [15, 16, 17, 18, 19, 20, 21]
    #     }
    # ]

    for prediction in predictions:
        station_number = prediction['station_number']

        # Get the corresponding CSV file
        filename = f"station{station_number}_data.csv"
        filepath = os.path.join(merged_data_dir, filename)
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Find the index of the row that matches the "datetime" from the prediction
        index = df[df['datetime'] == prediction['datetime']].index[0]

        # Check if there are "time_interval" rows after the found index
        if index + time_interval < len(df):
            ABS_scaler = joblib.load(os.path.join(models_dir, f'station{station_number}', 'ABS_scaler.pkl'))
            print(f"Station {station_number}:\n")
            mse, mae, ev = evaluate_predictions(df, index, station_number, ABS_scaler, prediction)
            print(f"MSE = {mse}, MAE = {mae}, EV = {ev}\n")
        else:
            print(f"Did not find {time_interval} rows after the prediction datetime for station {station_number}")


if __name__ == "__main__":
    main()