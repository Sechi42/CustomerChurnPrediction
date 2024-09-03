# Importing Dependencies
from flask import Flask, render_template, request
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

# Define the transform_to_integers function
def transform_to_integers(request_data):
    # Implement the logic to convert necessary fields to integer or float
    # Example:
    numeric_fields = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']
    for field in numeric_fields:
        if field in request_data:
            request_data[field] = float(request_data[field])
    return request_data

# Load the classification pipeline
classification_pipeline_4 = load_pipeline(pipeline_to_load=config.MODEL_NAME_4, model_name=config.MODEL_NAME_4)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("homepage.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        request_data = dict(request.form)
        print("Form data received:", request_data)
        request_data = transform_to_integers(request_data)
        print("Transformed data:", request_data)
        data = pd.DataFrame([request_data])
        print("DataFrame:", data)
        pred = classification_pipeline_4.predict(data)
        print(f"Prediction: {pred}")

        if int(pred[0]) == 1:
            result = "Your client probably will not renew the contract"
        else:
            result = "Your client probably will renew the contract"

        return render_template('homepage.html', prediction=result)

@app.errorhandler(500)
def internal_error(error):
    return "500: Something went wrong"

@app.errorhandler(404)
def not_found(error):
    return "404: Page not found", 404

if __name__ == "__main__":
    app.run(debug=True)
