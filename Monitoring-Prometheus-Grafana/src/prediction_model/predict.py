import pandas as pd
import numpy as np
import joblib
from prediction_model.config import config  # Ajustado para la estructura correcta
from prediction_model.processing.data_handling import load_pipeline, load_dataset
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Optional

# Cargar los pipelines de clasificación
classification_pipelines = {
    config.MODEL_NAME_1: load_pipeline(pipeline_to_load=config.MODEL_NAME_1, model_name=config.MODEL_NAME_1),
    config.MODEL_NAME_2: load_pipeline(pipeline_to_load=config.MODEL_NAME_2, model_name=config.MODEL_NAME_2),
    config.MODEL_NAME_3: load_pipeline(pipeline_to_load=config.MODEL_NAME_3, model_name=config.MODEL_NAME_3),
    config.MODEL_NAME_4: load_pipeline(pipeline_to_load=config.MODEL_NAME_4, model_name=config.MODEL_NAME_4),
}

def evaluate_model(classification_pipeline, model_name):
    test_data = load_dataset(config.TEST_FILE)
    print()
    print(f"Evaluating {model_name} with test data shape: {test_data.shape}")
    targets_test = test_data[config.TARGET].apply(lambda x: 0 if x == 'No' else 1)
    test_data = test_data.drop(['customerID', 'EndDate'], axis=1)
    
    probabilities = classification_pipeline.predict_proba(test_data)[:, 1]
    predictions = classification_pipeline.predict(test_data)
    
    auc_roc = roc_auc_score(targets_test, probabilities)
    print(f'{model_name} AUC-ROC: {auc_roc:.2f}')
    
    accuracy = accuracy_score(targets_test, predictions)
    print(f'{model_name} Accuracy: {accuracy:.2f}')
    
    return auc_roc, accuracy

def predict_single_instance(model_name, input_data: dict):
    input_df = pd.DataFrame([input_data])
    
    if 'customerID' in input_df.columns:
        input_df = input_df.drop(['customerID'], axis=1)
    
    pipeline = classification_pipelines[model_name]
    
    probabilities = pipeline.predict_proba(input_df)[:, 1]
    prediction = pipeline.predict(input_df)

    return {
        'prediction': int(prediction[0]),  
        'probability': probabilities[0]  
    }

if __name__ == '__main__':
    for model_name, pipeline in classification_pipelines.items():
        evaluate_model(classification_pipeline=pipeline, model_name=model_name)

