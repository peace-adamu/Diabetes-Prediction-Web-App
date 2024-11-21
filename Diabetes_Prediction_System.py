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