from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from typing import Optional
from pathlib import Path
import os
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.middleware.cors import CORSMiddleware
from src.prediction_model.config import config
from src.prediction_model.predict import predict_single_instance


# Configuración del puerto
port = int(os.environ.get("PORT", 8005))

# Inicializar la aplicación FastAPI
app = FastAPI(
    title="Churn Prediction App using API - CI CD Jenkins",
    description="A simple CI CD Demo",
    version='1.0'
)

# Configuración de CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

# Instrumentación de Prometheus
Instrumentator().instrument(app).expose(app)

# Definición del modelo de entrada usando Pydantic
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


# Ruta principal
@app.get('/')
def index():
    return {'message': "Welcome to Churn Prediction"}

# Ruta para predicción basada en datos JSON
@app.post('/predict')
def predict_churn_status(churn_details: ChurnPred):
    # Convertir datos de entrada a diccionario
    data = churn_details.dict()
    
    # Selección del modelo (se usa el modelo 4 como ejemplo)
    model_name = config.MODEL_NAME_4
    
    # Realizar predicción
    prediction_result = predict_single_instance(model_name, data)
    
    # Extraer probabilidad y resultado
    probability = float(prediction_result['probability'])
    pred = 'Yes' if prediction_result['prediction'] == 1 else 'No'
    
    return {
        'Status of Churn Application': pred,
        'Probability': probability
    }

# Ruta para predicción desde la UI o parámetros GET
@app.post("/prediction_ui")
def predict_gui(
    BeginDate: str,
    Type: str,
    PaperlessBilling: str,
    PaymentMethod: str,
    MonthlyCharges: float,
    TotalCharges: str,
    InternetService: Optional[str] = None,
    OnlineSecurity: Optional[str] = None,
    OnlineBackup: Optional[str] = None,
    DeviceProtection: Optional[str] = None,
    TechSupport: Optional[str] = None,
    StreamingTV: Optional[str] = None,
    StreamingMovies: Optional[str] = None,
    MultipleLines: Optional[str] = None,
    gender: str = None,
    SeniorCitizen: int = None,
    Partner: str = None,
    Dependents: str = None,
):
    # Convertir parámetros a diccionario
    data = {
        "BeginDate": BeginDate,
        "Type": Type,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "MultipleLines": MultipleLines,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
    }

    # Selección del modelo (usando el modelo 4 como ejemplo)
    model_name = config.MODEL_NAME_4
    prediction_result = predict_single_instance(model_name, data)

    # Extraer probabilidad y resultado
    probability = float(prediction_result['probability'])
    pred = 'Yes' if prediction_result['prediction'] == 1 else 'No'

    # Retornar el resultado de la predicción
    return {
        'Churn Prediction': pred,
        'Probability': probability
    }


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=port, reload=False)
