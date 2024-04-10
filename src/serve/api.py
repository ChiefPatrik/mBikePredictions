from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import os
import numpy as np
import pandas as pd
import joblib
import uvicorn
from typing import List
from tensorflow.keras.models import load_model

# Get the directory of the current Python script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the 'models' directory relative to the script's directory
models_dir = os.path.join(current_dir, '..', '..', 'models')

# Construct paths to individual model files within the 'models' directory
model_path = os.path.join(models_dir, 'prediction_model.h5')
scaler_path = os.path.join(models_dir, 'standard_scaler.pkl')

model = load_model(model_path)
standard_scaler = joblib.load(scaler_path)

# Specify request body for vaja4 & vaja6
class InputData(BaseModel):
    data: List[dict]

    @validator("data")
    def validate_data(cls, value):
        if not value or not all(isinstance(item, dict) for item in value):
            raise ValueError("Invalid data format. Expecting a non-empty list of dictionaries.")
        return value
    
def sortByDate(dataframe):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = dataframe.sort_values('date')
    return dataframe

# Create server
app = FastAPI()

@app.post("/mbajk/predict/", response_model=dict)
async def predict_mBike(data: InputData):
    try:
        input_data = pd.DataFrame(data.data)
        print(input_data)
        
        # Validate DataFrame columns and required attributes
        required_columns = ["date", "available_bike_stands"]
        missing_columns = set(required_columns) - set(input_data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Sort by date field
        bike_data = sortByDate(input_data)

        
        available_bike_stands = bike_data['available_bike_stands'].values.reshape(-1, 1)
        n_available_bike_stands = standard_scaler.transform(available_bike_stands)
        n_available_bike_stands = np.reshape(n_available_bike_stands, (n_available_bike_stands.shape[1], 1, n_available_bike_stands.shape[0]))

        prediction = model.predict(n_available_bike_stands)[0]

        inversed_prediction = standard_scaler.inverse_transform(np.array(prediction).reshape(-1, 1))
        return {"prediction": int(inversed_prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)