import streamlit as st
import pandas as pd
# import joblib  # Descomenta esto si vas a cargar un modelo entrenado

# Título de la aplicación
st.title("Streamlit Demo - SergioDS")

# Encabezados y texto
st.header("Header of Streamlit")
st.subheader("Sub-Heading of Streamlit")
st.text("This is an example text")

# Mensajes de estado
st.success("Success")
st.warning("Warning")
st.info("Information")
st.error("Error")

# Checkbox
if st.checkbox("Select/Unselect"):
    st.text("User selected the checkbox")  
else:
    st.text("User has not selected the checkbox")

# Radio buttons
state = st.radio("What is your favorite Color?", ("Red", "Green", 'Yellow'))
if state == "Green":
    st.text("That's my favorite color as well")

# Select box
occupation = st.selectbox("What do you do?", ['Student', 'Vlogger', "Engineer"])
st.text(f"Selected option is {occupation}")

# Botón de acción
if st.button("Example button"):
    st.error("You clicked it")

# Carga de archivo
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
else:
    st.text("No file uploaded yet.")

# Entrada de texto numérico
user_input = st.number_input("Enter a number", min_value=0, max_value=100, value=10)
st.write(f"User entered: {user_input}")

# Barra deslizante
slider_value = st.slider("Select a range of values", 0, 100, (25, 75))
st.write(f"Slider values: {slider_value}")

# Carga de un modelo y predicción (comentado para evitar errores)
# @st.cache_resource
# def load_model():
#     return joblib.load("model.pkl")

# if st.button("Make Prediction"):
#     model = load_model()
#     prediction = model.predict([[user_input]])
#     st.write(f"Model prediction: {prediction[0]}")
# else:
#     st.text("Prediction not made.")

# Gráficos (ejemplo básico)
if st.button("Show plot"):
    st.line_chart([1, 2, 3, 4, 5])

# Selección de múltiples opciones
options = st.multiselect("Choose some options", ["Option 1", "Option 2", "Option 3"])
st.write(f"Selected options: {options}")

# Barra lateral para navegación
st.sidebar.header("Sidebar Example")
sidebar_selection = st.sidebar.selectbox("Choose an option", ["Option A", "Option B"])
st.sidebar.write(f"You selected {sidebar_selection}")
