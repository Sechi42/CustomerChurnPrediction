import pandas as pd
import numpy as np 
from config import config  
from processing.data_handling import load_dataset,save_pipeline, split_dataset, merge_datasets
import processing.preprocessing as pp 
import pipeline as pipe 
import sys

def perform_training(model_name, pipeline):
    split_dataset()
    train_data = load_dataset(config.TRAIN_FILE)
    print(train_data.shape)
    targets_train = train_data[config.TARGET].apply(lambda x: 0 if x == 'No' else 1)
    train_data = train_data.drop(['customerID', 'EndDate'], axis=1)
    print(train_data.shape)
    print(targets_train)
    pipeline.fit(train_data,targets_train)
    save_pipeline(pipeline, model_name=model_name)

if __name__=='__main__':
    perform_training(config.MODEL_NAME_1, pipeline=pipe.classification_pipeline_LR)
    perform_training(config.MODEL_NAME_2, pipeline=pipe.classification_pipeline_RFC)
    perform_training(config.MODEL_NAME_3, pipeline=pipe.classification_pipeline_GB)
    perform_training(config.MODEL_NAME_4, pipeline=pipe.classification_pipeline_NN)
