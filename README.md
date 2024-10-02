# Project Title: Diabetes Prediction Web App
- [Project overview](#project-overview)
- [Features](#features)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Requirements](#requirements)
- [Visualization](#visualization)
- [Classification report & Confusion matrix](#Classification-report-&Confusion-matrix)
- [Code](#code)

  
## Overview
This project contains a web-based diabetes prediction system utilizing machine learning algorithms and Streamlit. The system predicts whether a patient has diabetes based on their medical characteristics. Gender, AGE, Urea, Cr, HbA1c, Chol, TG, HDL, LDL, VLDL, BMI, CLASS 

## Features  
- Data exploration and preprocessing
- Model training and testing using multiple algorithms:
- Logistic Regression
- K-Nearest Neighbors
- Decision Tree Classifier
- Random Forest Classifier
- Support Vector Machine
- Model evaluation using classification reports and confusion matrices
- Web-based deployment using Streamlit

## Usage
#### Installation
 Install dependencies: pip install -r requirements.txt
#### Running the App
Run the app: streamlit run app.py
#### Using the App
- Upload your dataset or use the provided sample dataset
- Select the model and hyperparameters
- Get predictions and visualize results

## Directory Structure
- models: Contains trained model files
- notebooks: Contains Jupyter notebooks for data exploration, model development, and testing
- app: Contains the Streamlit app code for deploying the model
- results: Contains results of model testing and evaluation

## Requirements
- Python 3.8+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Power BI

## Visualization

![g5](https://github.com/user-attachments/assets/a4355b1b-f822-4f37-b64b-5e8bf58434bd)

![g6](https://github.com/user-attachments/assets/9f6f9754-5179-4735-958f-23fb6318a366)


![g7](https://github.com/user-attachments/assets/36df2711-cfe9-4222-b13d-b9ecb495a338)


## Classification report & Confusion matrix
precision    recall  f1-score   support

           0       0.99      1.00      0.99       250
           1       1.00      1.00      1.00       239
           2       1.00      1.00      1.00       256
           3       1.00      0.98      0.99       257
           4       0.99      1.00      0.99       258

    
    
    accuracy                           1.00      1260
- macro avg       1.00      1.00      1.00      1260
- weighted avg       1.00      1.00      1.00      1260

- [[250   0   0   0   0]
-  [  0 239   0   0   0]
- [  0   0 256   0   0]
- [  3   0   0 251   3]
- [  0   0   0   0 258]]

 ## Code
 Python script code used
```
import streamlit as st
import joblib
import numpy as np


# Load the saved model with encoding
rfc_model = joblib.load(open('rfc_model.joblib', 'rb'))

# Function for prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = rfc_model.predict(input_data_reshaped)
    return prediction[0]  # Return the first element of the prediction array

# Main function
def main():
    # Title
    st.title('Diabetes Prediction Web App')

    # Input fields
    No_Pation = st.number_input('Number of patient', min_value=1000, max_value=99999, step=1)
    Age = st.number_input('Age of patient', min_value=1, max_value=120, step=1)
    Urea = st.number_input('Urea count', min_value=0.1, max_value=100.0, step=0.1)
    Cr = st.number_input('Creatinine count', min_value=0.1, max_value=10.0, step=0.1)
    HbAlc = st.number_input('HbAlc count', min_value=1.0, max_value=20.0, step=0.1)
    Chol = st.number_input('Cholesterol count', min_value=50.0, max_value=500.0, step=1.0)
    TG = st.number_input('TG count', min_value=10.0, max_value=500.0, step=1.0)
    HDL = st.number_input('HDL count', min_value=10.0, max_value=100.0, step=1.0)
    LDL = st.number_input('LDL count', min_value=10.0, max_value=200.0, step=1.0)
    VLDL = st.number_input('VLDL count', min_value=1.0, max_value=50.0, step=0.1)
    BMI = st.number_input('Body Mass Index value', min_value=10.0, max_value=50.0, step=0.1)
    Gender = st.selectbox('Gender', ['Female', 'Male'])

    # Convert gender to numerical value
    if Gender == 'Female':
        Gender = 0
    else:
        Gender = 1

    # Button for prediction
    if st.button('Diabetes Test Result'):
        try:
            diagnosis = diabetes_prediction([No_Pation, Age, Urea, Cr, HbAlc, Chol, TG, HDL, LDL, VLDL, BMI, Gender])
            if diagnosis == 0:
                st.success('The person is not diabetic')
            elif diagnosis == 1:
                st.success('The person is diabetic')
            elif diagnosis == 2:
                st.success('Predicted diabetic')
            else:
                st.success('Mixed prediction')
        except Exception as e:
            st.error(f"An error occurred: {e}")

        # Add a retry button
        st.button('Retry', key='retry')
        if st.session_state.get('retry'):
            st.experimental_rerun()

if __name__ == '__main__':
    main()
```


## Author
Peace Adamu
