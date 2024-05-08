import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

def load_model(model_path):
    # Load the Random Forest model from the .pkl file
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, features):
    # Make predictions using the loaded model
    prediction = model.predict(features)
    return prediction

def main():
    # Title of the web app
    st.title('Harnessing Technology for Farmer Welfare: Predictive Analytics for Rice Crop Yields')

    # Load the model
    model_path = 'random_forest_model.pkl'  # Replace with your model file path
    model = load_model(model_path)

    # Load CSV file containing feature names
    csv_file_path = 'Complete_data.csv'  # Replace with your CSV file path
    features_df = pd.read_csv(csv_file_path)
    st.sidebar.image('img.jpg', use_column_width=True)
    # Sidebar section for user input
    st.sidebar.title('User Input')

    # Collect user input for latitude and longitude
    latitude = st.sidebar.number_input('Latitude', value=0.0, format="%.3f")
    longitude = st.sidebar.number_input('Longitude', value=0.0, format="%.3f")

    # Find the row corresponding to the given latitude and longitude
    row = features_df[(features_df['Latitude'] == latitude) & (features_df['Longitude'] == longitude)]

    # Create a button to trigger prediction
    if st.sidebar.button('Predict Yield'):
        if not row.empty:
        # Extract features from the row
            features = row.values  # Assuming latitude and longitude are the first two columns

            # Prepare input for prediction
            input_features = features

            # Make prediction
            prediction = predict(model, input_features)

            # Display prediction
            st.write('## Prediction')
            st.write(f'Predicted value: {prediction[0]} quintal/acre')
        else:
            st.write('## No data found for the given latitude and longitude.')

if __name__ == '__main__':
    # Specify scikit-learn version
    main()