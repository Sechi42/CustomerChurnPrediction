import pathlib
import os


# PACKAGE_ROOT apunta a la carpeta 'project_16'
PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent

# DATAPATH apunta a la carpeta 'datasets' dentro de 'project_16'
DATAPATH = PACKAGE_ROOT / "datasets"

CONTRACT = 'contract.csv'
INTERNET = 'internet.csv'
PERSONAL = 'personal.csv'
PHONE = 'phone.csv'

TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'

MODEL_NAME_1 = 'logistic_classification.pkl'
MODEL_NAME_2 = 'randforst_classification.pkl'
MODEL_NAME_3 = 'gradboost_classification.pkl'
MODEL_NAME_4 = 'cnn_classification.pkl'

SAVE_MODEL_PATH = PACKAGE_ROOT / 'trained_models'

TARGET = 'EndDate'

POST_MERGE_COLUMNS = [
    'BeginDate', 'Type', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'customerID', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'MultipleLines', 'gender', 'SeniorCitizen', 'Partner', 'Dependents'
]

COLUMNS_TO_INTRODUCE = [
    'BeginDate', 'Type', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'MultipleLines', 'gender', 'SeniorCitizen', 'Partner', 'Dependents'
]

#Final features used in the model
FEATURES = ['type', 'paperless_billing', 'payment_method', 'internet_service', 'online_security', 
            'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
            'multiple_lines', 'partner', 'dependents', 'begin_year', 'begin_month', 'monthly_charges', 
            'total_charges', 'time_in_company', 'senior_citizen', 'begin_date', 'genre', 'customer_id']

FEATURES_TO_FULL = ['internet_service', 'online_security', 'online_backup', 'device_protection', 
                    'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines']

NUM_FEATURES = ['monthly_charges', 'total_charges', 'time_in_company', 'senior_citizen']

CAT_FEATURES = ['type', 'paperless_billing', 'payment_method', 'internet_service', 'online_security', 
            'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
            'multiple_lines', 'partner', 'dependents', 'begin_year', 'begin_month']

# in our case it is same as Categorical features
FEATURES_TO_ENCODE = ['type', 'paperless_billing', 'payment_method', 'internet_service', 'online_security', 
            'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
            'multiple_lines', 'partner', 'dependents', 'begin_year', 'begin_month']

FEATURE_TO_MODIFY = 'begin_date'

FEATURE_TO_ADD = ['begin_month', 'begin_year', 'time_in_company']

NEW_COLUMN = 'time_in_company'

DROP_FEATURES = ['begin_date', 'gender']

FEATURES_TO_NUMERIC = 'total_charges'

FEATURES_TO_CALCULATE = 'total_charges'

FEATURES_BEGIN_MONTH = 'begin_month'

FEATURES_BEGIN_YEAR = 'begin_year'

FEATURE_TO_MASK = 'type'

FEATURE_MONTH = 'monthly_charges'

FEATURE_TO_ORDER = 'begin_date'

FEATURE_TOTAL_DAYS = 'time_in_company'

FEATURES_TO_DATETIME = 'begin_date'

FEATURES_BEGIN_DATE = 'begin_date'

FINAL_FEATURES = ['type', 'paperless_billing', 'payment_method', 'internet_service', 'online_security', 
            'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 
            'multiple_lines', 'partner', 'dependents', 'begin_year', 'begin_month', 'monthly_charges', 
            'total_charges', 'time_in_company', 'senior_citizen']
