import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import numpy as np


#Librerias para red neuronal

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import os



class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.0001, epochs=100, batch_size=64, patience=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.model = None
        self.history = None
        self.input_dim = None  # Se configurará dinámicamente

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=512, activation='relu', input_dim=self.input_dim, kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(units=256, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(units=128, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(units=64, activation='relu', kernel_regularizer='l2'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(units=1, activation='sigmoid'))
        optimizer = AdamW(learning_rate=self.learning_rate)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['AUC'])

        return model

    def fit(self, X, y, X_val=None, y_val=None):
        self.input_dim = X.shape[1]  # Configurar input_dim basado en el tamaño de X
        self.model = self.build_model()

        early_stopping = EarlyStopping(monitor='AUC', patience=self.patience, restore_best_weights=True)

        if X_val is not None and y_val is not None:
            self.history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=1
            )
        else:
            self.history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                callbacks=[early_stopping],
                verbose=1
            )

        return self

    def predict_proba(self, X):
        probabilities = self.model.predict(X).flatten()
        return np.vstack([1 - probabilities, probabilities]).T

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities > 0.5).astype(int)


# 1. ColumnNameChanger
class ColumnNameChanger:
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # No necesita hacer nada para "ajustar" los datos
        return self

    def transform(self, df):
        df = df.copy()  # Crear una copia del DataFrame
        new_cols = []
        for col_name in df.columns:
            new_col_name = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', col_name)
            new_col_name = new_col_name.lower()
            new_col_name = re.sub(r'_+', '_', new_col_name).strip('_')
            new_cols.append(new_col_name)
        df.columns = new_cols
        
        return df

# 2. ColumnsToNumericAndDatetime
class ColumnsToNumericAndDatetime:
    def __init__(self, variables_to_numeric, variables_to_datetime):
        self.variables_to_numeric = variables_to_numeric
        self.variables_to_datetime = variables_to_datetime
        
    def fit(self, X, y=None):
        # No hacer nada si no es necesario ajustar datos
        return self

    def transform(self, df):
        df = df.copy()  # Crear una copia del DataFrame
        
        # Convertir a numérico
        df[self.variables_to_numeric] = df[self.variables_to_numeric].apply(pd.to_numeric, errors='coerce')
        
        # Convertir a datetime
        df[self.variables_to_datetime] = df[self.variables_to_datetime].apply(pd.to_datetime, errors='coerce')
        
        return df

# 3. TotalChargeCalculate
class TotalChargeCalculate:
    def __init__(self, variables_to_calculate, variables_to_mask, month_variable):
        self.variables_to_calculate = variables_to_calculate
        self.variables_to_mask = variables_to_mask
        self.month_variable = month_variable
        
    def fit(self, X, y=None):
        # No hacer nada si no es necesario ajustar datos
        return self

    def transform(self, df):
        df = df.copy()  # Crear una copia del DataFrame
        df[self.variables_to_mask] = df[self.variables_to_mask].str.lower()
        mask_two_year = (df[self.variables_to_calculate].isnull()) & (df[self.variables_to_mask] == 'two year')
        mask_one_year = (df[self.variables_to_calculate].isnull()) & (df[self.variables_to_mask] == 'one year')

        df.loc[mask_two_year, self.variables_to_calculate] = df.loc[mask_two_year, self.month_variable] * 24
        df.loc[mask_one_year, self.variables_to_calculate] = df.loc[mask_one_year, self.month_variable] * 12
        return df

# 4. SortDataset
class SortDataset:
    def __init__(self, variables_to_order):
        self.variables_to_order = variables_to_order
    
    def fit(self, X, y=None):
        # No hacer nada si no es necesario ajustar datos
        return self

    def transform(self, df):
        df = df.copy()  # Crear una copia del DataFrame
        return df.sort_values(by=self.variables_to_order)

# 5. UnknownImputer
class UnknownImputer:
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        # No hacer nada si no es necesario ajustar datos
        return self

    def transform(self, df):
        df = df.copy()  # Crear una copia del DataFrame
        df[self.variables] = df[self.variables].fillna('unknown')
        return df

# 6. CalculateTotalDays
class CalculateTotalDays:
    def __init__(self, variables_to_calculate, variable_begin_date):
        self.variables_to_calculate = variables_to_calculate
        self.variable_begin_date = variable_begin_date

    def fit(self, X, y=None):
        # No hacer nada si no es necesario ajustar datos
        return self

    def transform(self, df):
        df = df.copy()  # Crear una copia del DataFrame
        fecha_actual = df[self.variable_begin_date].max()
        df[self.variables_to_calculate] = (fecha_actual - df[self.variable_begin_date]).dt.days
        return df

# 7. CalculateMonthsYearsClass
class CalculateMonthsYearsClass:
    def __init__(self, variables_begin_date, variable_begin_month, variable_begin_year):
        self.variables_begin_date = variables_begin_date
        self.variable_begin_month = variable_begin_month
        self.variable_begin_year = variable_begin_year
        
    def fit(self, X, y=None):
        # No hacer nada si no es necesario ajustar datos
        return self

    def transform(self, df):
        df = df.copy()  # Crear una copia del DataFrame
        df[self.variable_begin_year] = df[self.variables_begin_date].dt.year
        df[self.variable_begin_month] = df[self.variables_begin_date].dt.month
        return df

# 8. DropColumns
class DropColumns:
    def __init__(self, variables_to_drop):
        self.variables_to_drop = variables_to_drop
        
    def fit(self, X, y=None):
        # No hacer nada si no es necesario ajustar datos
        return self

    def transform(self, X):
        X = X.copy()  # Crear una copia del DataFrame
        X = X.drop(self.variables_to_drop, axis=1)
        return X

# 9. CustomOHETransformer
class CustomOHETransformer:
    def __init__(self, columns):
        self.columns = columns
        self.encoder = None
        
    def fit(self, X, y=None):
        # Crear el codificador OneHotEncoder y ajustarlo a las columnas categóricas
        self.encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        self.encoder.fit(X[self.columns])
        return self

    def transform(self, X):
        X = X.copy()  # Crear una copia del DataFrame
        # Aplicar la codificación OneHot a las columnas categóricas
        encoded_columns = self.encoder.transform(X[self.columns])
        # Convertir las columnas codificadas en un DataFrame
        encoded_df = pd.DataFrame(encoded_columns, columns=self.encoder.get_feature_names_out(self.columns))
        # Concatenar las columnas codificadas con las otras columnas del DataFrame
        X = X.drop(columns=self.columns).reset_index(drop=True)
        X = pd.concat([X, encoded_df], axis=1)
        return X

# 10. CustomScaler
class CustomScaler:
    def __init__(self, columns):
        self.columns = columns
        self.scaler = None

    def fit(self, X, y=None):
        # Verificar que las columnas existen en X
        if not set(self.columns).issubset(X.columns):
            raise ValueError("Algunas columnas especificadas no están en el DataFrame.")
        
        # Crear y ajustar el escalador solo a las columnas numéricas
        self.scaler = StandardScaler()
        self.scaler.fit(X[self.columns])
        return self

    def transform(self, X):
        # Verificar que las columnas existen en X
        if not set(self.columns).issubset(X.columns):
            raise ValueError("Algunas columnas especificadas no están en el DataFrame.")
        
        # Aplicar la normalización a las columnas numéricas
        scaled_columns = self.scaler.transform(X[self.columns])
        
        # Convertir las columnas escaladas en un array numpy
        scaled_array = np.array(scaled_columns)
        
        # Eliminar las columnas escaladas del DataFrame original y convertir el resto en array numpy
        X_dropped = X.drop(columns=self.columns)
        non_scaled_array = np.array(X_dropped)
        
        # Concatenar el array escalado y el array de las columnas no escaladas
        final_array = np.concatenate([scaled_array, non_scaled_array], axis=1)

        return final_array