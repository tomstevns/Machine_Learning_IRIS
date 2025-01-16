import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


# Function to load data from file
def load_data(filename):
    try:
        # Load the dataset
        data = pd.read_csv(filename, sep=';', header=0)
        return data
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        exit()


# Function to process data
def process_data(data):
    # Encode the Population column
    le = LabelEncoder()
    data['Population'] = le.fit_transform(data['Population'])

    # Split the dataset into features and labels
    X = data[['Sample', 'Barcode']]
    y = data['Population']

    # Encode categorical features (Sample and Barcode)
    X_encoded = pd.get_dummies(X, columns=['Sample', 'Barcode'])

    return X_encoded, y, le


# Function to train and save model
def train_and_save_model(X_train, y_train, model_filename):
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the model
    with open(model_filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_filename}")


# Main function
def main():
    # Input filename from the console
    # filename = input("Enter the path to the dataset file: ")
    filename = "AnnotationCells.csv"
    model_filename = "population_model.pkl"

    # Load data
    data = load_data(filename)

    # Process data
    X, y, label_encoder = process_data(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and save the model
    train_and_save_model(X_train, y_train, model_filename)

    # Save the label encoder
    encoder_filename = "label_encoder.pkl"
    with open(encoder_filename, 'wb') as file:
        pickle.dump(label_encoder, file)
    print(f"Label encoder saved to {encoder_filename}")


if __name__ == "__main__":
    main()
