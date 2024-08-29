from sklearn.pipeline import Pipeline
from processing import preprocessing as pp
from config import config 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os

classification_pipeline_LR = Pipeline(
    [
        ('ChangeNameColumns', pp.ColumnNameChanger()),
        ('ToNumericAndDatetimeColumns', pp.ColumnsToNumericAndDatetime(
            variables_to_numeric=config.FEATURES_TO_NUMERIC, 
            variables_to_datetime=config.FEATURES_TO_DATETIME)),
        ('TotalChargeCalculate', pp.TotalChargeCalculate(variables_to_calculate=config.FEATURES_TO_CALCULATE, 
                                                      variables_to_mask=config.FEATURE_TO_MASK, 
                                                      month_variable=config.FEATURE_MONTH)),
        ('UnknownImputation', pp.UnknownImputer(variables=config.FEATURES_TO_FULL)),
        ('TimeInCompanyCalculate', pp.CalculateTotalDays(variables_to_calculate=config.FEATURE_TOTAL_DAYS, 
                                                      variable_begin_date=config.FEATURES_BEGIN_DATE)),
        ('CalculateNewVariables', pp.CalculateMonthsYearsClass(
                                                            variables_begin_date=config.FEATURES_BEGIN_DATE,
                                                            variable_begin_month=config.FEATURES_BEGIN_MONTH,
                                                            variable_begin_year=config.FEATURES_BEGIN_YEAR)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('OHEncoder', pp.CustomOHETransformer(columns=config.CAT_FEATURES)),
        ('StandarScaler', pp.CustomScaler(columns=config.NUM_FEATURES)),
        ('LogisticClassifier', LogisticRegression(random_state=12345))
    ]
)
classification_pipeline_RFC = Pipeline(
    [
        ('ChangeNameColumns', pp.ColumnNameChanger()),
        ('ToNumericAndDatetimeColumns', pp.ColumnsToNumericAndDatetime(
            variables_to_numeric=config.FEATURES_TO_NUMERIC, 
            variables_to_datetime=config.FEATURES_TO_DATETIME)),
        ('TotalChargeCalculate', pp.TotalChargeCalculate(variables_to_calculate=config.FEATURES_TO_CALCULATE, 
                                                      variables_to_mask=config.FEATURE_TO_MASK, 
                                                      month_variable=config.FEATURE_MONTH)),
        ('UnknownImputation', pp.UnknownImputer(variables=config.FEATURES_TO_FULL)),
        ('TimeInCompanyCalculate', pp.CalculateTotalDays(variables_to_calculate=config.FEATURE_TOTAL_DAYS, 
                                                      variable_begin_date=config.FEATURES_BEGIN_DATE)),
        ('CalculateNewVariables', pp.CalculateMonthsYearsClass(
                                                            variables_begin_date=config.FEATURES_BEGIN_DATE,
                                                            variable_begin_month=config.FEATURES_BEGIN_MONTH,
                                                            variable_begin_year=config.FEATURES_BEGIN_YEAR)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('OHEncoder', pp.CustomOHETransformer(columns=config.CAT_FEATURES)),
        ('StandarScaler', pp.CustomScaler(columns=config.NUM_FEATURES)),
        ('RandomForestClassifier', RandomForestClassifier(random_state=12345))
    ]
)

classification_pipeline_GB = Pipeline(
    [
        ('ChangeNameColumns', pp.ColumnNameChanger()),
        ('ToNumericAndDatetimeColumns', pp.ColumnsToNumericAndDatetime(
            variables_to_numeric=config.FEATURES_TO_NUMERIC, 
            variables_to_datetime=config.FEATURES_TO_DATETIME)),
        ('TotalChargeCalculate', pp.TotalChargeCalculate(variables_to_calculate=config.FEATURES_TO_CALCULATE, 
                                                      variables_to_mask=config.FEATURE_TO_MASK, 
                                                      month_variable=config.FEATURE_MONTH)),
        ('UnknownImputation', pp.UnknownImputer(variables=config.FEATURES_TO_FULL)),
        ('TimeInCompanyCalculate', pp.CalculateTotalDays(variables_to_calculate=config.FEATURE_TOTAL_DAYS, 
                                                      variable_begin_date=config.FEATURES_BEGIN_DATE)),
        ('CalculateNewVariables', pp.CalculateMonthsYearsClass(
                                                            variables_begin_date=config.FEATURES_BEGIN_DATE,
                                                            variable_begin_month=config.FEATURES_BEGIN_MONTH,
                                                            variable_begin_year=config.FEATURES_BEGIN_YEAR)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('OHEncoder', pp.CustomOHETransformer(columns=config.CAT_FEATURES)),
        ('StandarScaler', pp.CustomScaler(columns=config.NUM_FEATURES)),
        ('GradientBoostingClassifier', GradientBoostingClassifier(random_state=12345))
    ]
)



classification_pipeline_NN = Pipeline(
    [
        ('ChangeNameColumns', pp.ColumnNameChanger()),
        ('ToNumericAndDatetimeColumns', pp.ColumnsToNumericAndDatetime(
            variables_to_numeric=config.FEATURES_TO_NUMERIC, 
            variables_to_datetime=config.FEATURES_TO_DATETIME)),
        ('TotalChargeCalculate', pp.TotalChargeCalculate(variables_to_calculate=config.FEATURES_TO_CALCULATE, 
                                                      variables_to_mask=config.FEATURE_TO_MASK, 
                                                      month_variable=config.FEATURE_MONTH)),
        ('UnknownImputation', pp.UnknownImputer(variables=config.FEATURES_TO_FULL)),
        ('TimeInCompanyCalculate', pp.CalculateTotalDays(variables_to_calculate=config.FEATURE_TOTAL_DAYS, 
                                                      variable_begin_date=config.FEATURES_BEGIN_DATE)),
        ('CalculateNewVariables', pp.CalculateMonthsYearsClass(
                                                            variables_begin_date=config.FEATURES_BEGIN_DATE,
                                                            variable_begin_month=config.FEATURES_BEGIN_MONTH,
                                                            variable_begin_year=config.FEATURES_BEGIN_YEAR)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('OHEncoder', pp.CustomOHETransformer(columns=config.CAT_FEATURES)),
        ('StandarScaler', pp.CustomScaler(columns=config.NUM_FEATURES)),
        ('NeuralNetworkClassifier', pp.NeuralNetworkClassifier())  # Aquí usamos el clasificador de la red neuronal
    ]
)