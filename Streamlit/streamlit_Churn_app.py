import joblib
import streamlit as st 
import pandas as pd
import sys
import os
from pathlib import Path

# # Adding the below path to avoid module not found error
PACKAGE_ROOT =  Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# # Then perform import
from config import config
from processing.data_handling import load_pipeline

classification_pipeline_4 = load_pipeline(pipeline_to_load=config.MODEL_NAME_4, model_name=config.MODEL_NAME_4)

# Estilos personalizados
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
    }
    h1 {
        color: #4a76a8;
        text-align: center;
    }
    .stButton>button {
        background-color: #4a76a8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def prediction(BeginDate, Type, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
    MultipleLines, gender, SeniorCitizen, Partner, Dependents):
    
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
        "Dependents": Dependents
    }
    
    df = pd.DataFrame(data, index=[0])
    
    try:
        prediction = classification_pipeline_4.predict(df)
        st.write(f"Prediction raw output: {prediction}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Error during prediction"
    
    if prediction[0] == 0:
        pred = "The client possibly is going to continue"
    else:
        pred = "The client possibly is not going to continue"
        
    return pred    


def main():
    # Título y descripción
    st.title("Welcome to ChurnCustomers App")
    st.markdown("---")  # Línea divisoria

    st.header("Please enter the details of your client")
    
    # Sección 1: Datos básicos
    st.subheader("Basic Information")
    
    col1, col2 = st.columns(2)
    with col1:
        BeginDate = st.date_input("Enter the start date of the contract", value=None)
        if BeginDate:
            BeginDate = BeginDate.strftime('%Y/%m/%d')
            st.write("Selected date:", BeginDate)
        else:
            st.warning("Please select a start date.")
            return
    
    with col2:
        gender = st.selectbox('Gender', ("Male", "Female", "unknown"))
    
    # Sección 2: Información del cliente
    st.subheader("Customer Details")
    
    Type = st.selectbox('Contract Type', ("Month-to-month", "One year", "Two year"))
    PaperlessBilling = st.selectbox('Paperless Billing?', ("Yes", "No"))
    PaymentMethod = st.selectbox('Payment Method', 
                                 ("Mailed check", "Electronic check", "Credit card (automatic)", "Bank transfer (automatic)"))
    
    # Sección 3: Información del servicio
    st.subheader("Service Information")
    
    col1, col2 = st.columns(2)
    with col1:
        MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, format="%.2f")
        TotalCharges = st.number_input('Total Charges', min_value=0.0, format="%.2f")
    
    with col2:
        InternetService = st.selectbox('Internet Service', ("DSL", "Fiber optic", "No", "unknown"))
        OnlineSecurity = st.selectbox('Online Security', ("Yes", "No", "unknown"))
    
    # Sección 4: Otros servicios
    st.subheader("Additional Services")
    
    col1, col2 = st.columns(2)
    with col1:
        OnlineBackup = st.selectbox('Online Backup', ("Yes", "No", "unknown"))
        DeviceProtection = st.selectbox('Device Protection', ("Yes", "No", "unknown"))
        TechSupport = st.selectbox('Tech Support', ("Yes", "No", "unknown"))
    
    with col2:
        StreamingTV = st.selectbox('Streaming TV', ("Yes", "No", "unknown"))
        StreamingMovies = st.selectbox('Streaming Movies', ("Yes", "No", "unknown"))
        MultipleLines = st.selectbox('Multiple Lines', ("Yes", "No", "unknown"))
    
    # Sección 5: Detalles adicionales
    st.subheader("Additional Details")
    
    SeniorCitizen = st.selectbox('Senior Citizen?', (0, 1))
    Partner = st.selectbox('Partner?', ("Yes", "No"))
    Dependents = st.selectbox('Dependents?', ("Yes", "No"))
    
    # Botón de predicción
    if st.button("Predict"):
        result = prediction(BeginDate, Type, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges, InternetService, OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
    MultipleLines, gender, SeniorCitizen, Partner, Dependents)
        
        if result == "The client possibly is going to continue":
            st.success("The client possibly is going to continue")
        else:
            st.error("The client possibly is not going to continue")

if __name__ == "__main__":
    main()
