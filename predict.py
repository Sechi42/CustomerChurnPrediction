import pandas as pd
import numpy as np
import joblib
from config import config  
from processing.data_handling import load_pipeline, load_dataset, split_dataset
import os
from sklearn.metrics import roc_auc_score, accuracy_score

# Cargar los pipelines de clasificación
classification_pipelines = {
    config.MODEL_NAME_1: load_pipeline(pipeline_to_load=config.MODEL_NAME_1, model_name=config.MODEL_NAME_1),
    config.MODEL_NAME_2: load_pipeline(pipeline_to_load=config.MODEL_NAME_2, model_name=config.MODEL_NAME_2),
    config.MODEL_NAME_3: load_pipeline(pipeline_to_load=config.MODEL_NAME_3, model_name=config.MODEL_NAME_3),
    config.MODEL_NAME_4: load_pipeline(pipeline_to_load=config.MODEL_NAME_4, model_name=config.MODEL_NAME_4),
}

def evaluate_model(classification_pipeline, model_name):
    # Cargar los datos de prueba
    test_data = load_dataset(config.TEST_FILE)
    print()
    print(f"Evaluating {model_name} with test data shape: {test_data.shape}")
    targets_test = test_data[config.TARGET].apply(lambda x: 0 if x == 'No' else 1)
    test_data = test_data.drop(['customerID', 'EndDate'], axis=1)
    
    # Obtener las probabilidades predichas
    probabilities = classification_pipeline.predict_proba(test_data)[:, 1]

    # Obtener las predicciones binarias
    predictions = classification_pipeline.predict(test_data)
    
    # Calcular AUC-ROC usando las probabilidades
    auc_roc = roc_auc_score(targets_test, probabilities)
    print(f'{model_name} AUC-ROC: {auc_roc:.2f}')
    
    # Calcular Accuracy usando las predicciones binarias
    accuracy = accuracy_score(targets_test, predictions)
    print(f'{model_name} Accuracy: {accuracy:.2f}')
    
    return auc_roc, accuracy

if __name__ == '__main__':
    for model_name, pipeline in classification_pipelines.items():
        evaluate_model(classification_pipeline=pipeline, model_name=model_name)
