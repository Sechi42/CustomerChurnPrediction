# Importing dependencies

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Optional

# # Adding the below path to avoid module not found error

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# Then perform import

from config import config
from processing.data_handling import load_dataset, load_pipeline, merge_datasets

classification_pipeline_4 = load_pipeline(pipeline_to_load=config.MODEL_NAME_4, model_name=config.MODEL_NAME_4)

app = FastAPI()


# Perform parsing

class ChurnPred(BaseModel):
    BeginDate: str
    Type: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    MultipleLines: Optional[str] = None
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str

@app.get('/')
def index():
    return {'message': "Welcome to Churn Prediction"}


# defining the function which will make the prediction using the data which the user inputs
@app.post('/predict')
def predict_churn_status(churn_details: ChurnPred):
    data = churn_details.model_dump()
    new_data = {
        "BeginDate": data['BeginDate'],
        "Type": data['Type'],
        "PaperlessBilling": data['PaperlessBilling'],
        "PaymentMethod": data['PaymentMethod'],
        "MonthlyCharges": data['MonthlyCharges'],
        "TotalCharges": data['TotalCharges'],
        "InternetService": data.get('InternetService', None),
        "OnlineSecurity": data.get('OnlineSecurity', None),
        "OnlineBackup": data.get('OnlineBackup', None),
        "DeviceProtection": data.get('DeviceProtection', None),
        "TechSupport": data.get('TechSupport', None),
        "StreamingTV": data.get('StreamingTV', None),
        "StreamingMovies": data.get('StreamingMovies', None),
        "MultipleLines": data.get('MultipleLines', None),
        "gender": data['gender'],
        "SeniorCitizen": data['SeniorCitizen'],
        "Partner": data['Partner'],
        "Dependents": data['Dependents']
    }
# Create a DataFrame with a sigle row from the new_data dictionary
    df = pd.DataFrame([new_data])
    
    # Making predicitons
    
    prediction = classification_pipeline_4.predict(df)
    
    if prediction[0] == 0:
        pred = 'No'
    else:
        pred = 'Yes'
        
    return {'Status of Churn Application':pred}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)