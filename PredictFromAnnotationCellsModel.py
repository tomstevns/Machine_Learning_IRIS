import pandas as pd
import pickle
import numpy as np


# Function to load model and encoder
def load_model_and_encoder(model_filename, encoder_filename):
    try:
        # Load the trained model
        with open(model_filename, 'rb') as model_file:
            model = pickle.load(model_file)

        # Load the label encoder
        with open(encoder_filename, 'rb') as encoder_file:
            label_encoder = pickle.load(encoder_file)

        return model, label_encoder
    except FileNotFoundError as e:
        print(e)
        exit()


# Function to predict population
def predict_population(model, label_encoder):
    try:
        # Input data from the console
        sample = input("Enter Sample: ")
        barcode = input("Enter Barcode: ")

        # Encode input data (similar to training data processing)
        data = pd.DataFrame([[sample, barcode]], columns=['Sample', 'Barcode'])
        data_encoded = pd.get_dummies(data, columns=['Sample', 'Barcode'])

        # Align with the model's training data (fill missing columns with zeros)
        all_columns = model.feature_names_in_
        data_encoded = data_encoded.reindex(columns=all_columns, fill_value=0)

        # Make prediction
        prediction = model.predict(data_encoded)
        predicted_class = label_encoder.inverse_transform(prediction)[0]

        print(f"Predicted Population: {predicted_class}")
    except Exception as e:
        print(f"An error occurred during prediction: {e}")


# Main function
def main():
    # Filenames for the model and encoder
    model_filename = "population_model.pkl"
    encoder_filename = "label_encoder.pkl"

    # Load model and encoder
    model, label_encoder = load_model_and_encoder(model_filename, encoder_filename)

    # Predict population based on user input
    predict_population(model, label_encoder)


if __name__ == "__main__":
    main()
