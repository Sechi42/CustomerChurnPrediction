import os
import pandas as pd
import joblib
from config import config
from functools import reduce
from sklearn.model_selection import train_test_split

#Load the dataset
def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    return _data


# merge de los datasets
def merge_datasets():
    # Carga de datasets
    df_contract = load_dataset(config.CONTRACT)
    df_internet = load_dataset(config.INTERNET)
    df_personal = load_dataset(config.PERSONAL)
    df_phone = load_dataset(config.PHONE)
    # merging
    dfs = [df_contract, df_internet, df_phone, df_personal]
    df_merged = reduce(lambda left, right: pd.merge(left, right, on='customerID', how='outer'), dfs)
    
    return df_merged

# split dataset
def split_dataset():
    df = merge_datasets()
    train_data, test_data = train_test_split(df, random_state=12345)
    
    train_file = os.path.join(config.DATAPATH, 'train.csv')
    test_file = os.path.join(config.DATAPATH, 'test.csv')
    
    test_data.to_csv(test_file, index=False)
    train_data.to_csv(train_file, index=False)
    
    print(f"Los datasets se han guardado en:\n{train_file}\n{test_file}")
    
#Serialization
def save_pipeline(pipeline_to_save, model_name):
    save_path = os.path.join(config.SAVE_MODEL_PATH,model_name)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Model has been saved under the name {model_name}")

#Deserialization
def load_pipeline(pipeline_to_load, model_name):
    save_path = os.path.join(config.SAVE_MODEL_PATH,model_name)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded