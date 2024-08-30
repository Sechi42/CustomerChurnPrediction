import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

# Adding the below path to avoid module not found error
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# Then perform import
from config import config
from processing.data_handling import load_pipeline

# Load the classification pipeline
classification_pipeline_4 = load_pipeline(pipeline_to_load=config.MODEL_NAME_4, model_name=config.MODEL_NAME_4)

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error(f"El archivo {file_name} no se encontró.")


# Load the CSS file
load_css("styles.css")

st.markdown("---")  # Divider line

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
    
    st.title("Welcome to ChurnCustomers App")
    
    st.markdown("""
    <div class="description">
    Interconnect, a telecommunications operator, is aiming to predict customer churn rates. 
    By identifying customers likely to leave, the company plans to offer promotional codes and special plan options to retain them. 
    The marketing team has collected various customer data, including contract and plan information.

    This app uses a neural network model to predict whether a customer is likely to continue or not based on their input information.
    Please enter the details below to get a prediction.
    </div>
    """, unsafe_allow_html=True)
    
    st.header("Please enter the details of your client")
    
    st.markdown("---")
    
    # Sección 1: Datos básicos
    st.subheader("Basic Information")
    section_1_complete = False
    
    col1, col2 = st.columns(2)
    with col1:
        BeginDate = st.date_input("Enter the start date of the contract", value=None)
        if BeginDate:
            BeginDate = BeginDate.strftime('%Y/%m/%d')
            section_1_complete = True
        else:
            st.warning("Please select a start date.")
    
    if section_1_complete:
        with col2:
            gender = st.selectbox('Gender', ("Select", "Male", "Female", "unknown"), index=0)
            if gender != "Select":
                section_1_complete = True
            else:
                st.warning("Please select a gender.")
                section_1_complete = False
    st.markdown("---")
    # Mostrar la siguiente sección solo si la primera está completa
    if section_1_complete:
        # Sección 2: Información del cliente
        st.subheader("Customer Details")
        section_2_complete = False
        
        Type = st.selectbox('Contract Type', ("Select" ,"Month-to-month", "One year", "Two year"), index=0)
        PaperlessBilling = st.selectbox('Paperless Billing?', ("Select" ,"Yes", "No", "unknown"), index=0)
        PaymentMethod = st.selectbox('Payment Method', 
                                     ("Select" ,"Mailed check", "Electronic check", "Credit card (automatic)", "Bank transfer (automatic)"), index=0)
        
        if Type != "Select" and PaperlessBilling != "Select" and PaymentMethod != "Select":
            section_2_complete = True
        else:
            st.warning("Please select the contract type, payment method and paperless billing options.")
        st.markdown("---")
        # Mostrar la siguiente sección solo si la segunda está completa
        if section_2_complete:
            # Sección 3: Información del servicio
            st.subheader("Service Information")
            section_3_complete = False
            
            col1, col2 = st.columns(2)
            with col1:
                MonthlyCharges = st.number_input('Monthly Charges', min_value=0.0, format="%.2f")
                TotalCharges = st.number_input('Total Charges', min_value=0.0, format="%.2f")
            
            with col2:
                InternetService = st.selectbox('Internet Service', ("Select","DSL", "Fiber optic", "No", "unknown"), index=0)
                OnlineSecurity = st.selectbox('Online Security', ("Select" ,"Yes", "No", "unknown"), index=0)
                
            if InternetService != "Select" and OnlineSecurity != "Select" and MonthlyCharges > 0 and TotalCharges > 0:
                section_3_complete = True
            else:
                st.warning("Please complete all fields in Service Information.")
            st.markdown("---")
            # Mostrar la siguiente sección solo si la tercera está completa
            if section_3_complete:
                # Sección 4: Otros servicios
                st.subheader("Additional Services")
                section_4_complete = False
                
                col1, col2 = st.columns(2)
                with col1:
                    OnlineBackup = st.selectbox('Online Backup', ("Select" ,"Yes", "No", "unknown"), index=0)
                    DeviceProtection = st.selectbox('Device Protection', ("Select" ,"Yes", "No", "unknown"), index=0)
                    TechSupport = st.selectbox('Tech Support', ("Select" ,"Yes", "No", "unknown"), index=0)
                
                with col2:
                    StreamingTV = st.selectbox('Streaming TV', ("Select" ,"Yes", "No", "unknown"), index=0)
                    StreamingMovies = st.selectbox('Streaming Movies', ("Select" ,"Yes", "No", "unknown"), index=0)
                    MultipleLines = st.selectbox('Multiple Lines', ("Select" ,"Yes", "No", "unknown"), index=0)
                
                if OnlineBackup != "Select" and DeviceProtection != "Select" and TechSupport != "Select" and StreamingTV != "Select" and StreamingMovies != "Select" and MultipleLines != "Select":
                    section_4_complete = True
                else:
                    st.warning("Please complete all fields in Additional Services.")
                    
                    
                st.markdown("---")
                
                # Mostrar la siguiente sección solo si la cuarta está completa
                if section_4_complete:
                    # Sección 5: Detalles adicionales
                    st.subheader("Additional Details")
                    
                    SeniorCitizen = st.selectbox('Senior Citizen?', ("Select", 0, 1), index=0)
                    Partner = st.selectbox('Partner?', ("Select", "Yes", "No"), index=0)
                    Dependents = st.selectbox('Dependents?', ("Select", "Yes", "No"), index=0)
                    
                    if SeniorCitizen != "Select" and Partner != "Select" and Dependents != "Select":
                        if st.button("Predict"):
                            result = prediction(BeginDate, Type, PaperlessBilling, PaymentMethod,
                                                MonthlyCharges, TotalCharges, InternetService, OnlineSecurity,
                                                OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                                                MultipleLines, gender, SeniorCitizen, Partner, Dependents)
                            
                            if result == "The client possibly is going to continue":
                                st.success("The client possibly is going to continue")
                            else:
                                st.error("The client possibly is not going to continue")
                    else:
                        st.warning("Please complete all fields in Additional Details before prediction.")

if __name__ == "__main__":
    main()
