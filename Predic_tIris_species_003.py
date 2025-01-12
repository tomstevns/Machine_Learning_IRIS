# input data like: 5, 5, 1, 0.2   Sectosa
# input data like: 6, 3, 4, 1     Versicolor
# input data like: 6, 3 ,5, 2     Virginica
# Import libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load the Iris dataset (only for label encoding)
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=columns)

# Encode species column for decoding predictions
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# Define the path to the saved model
model_filename = "iris_model.pkl"

if not os.path.exists(model_filename):
    print(f"Model file '{model_filename}' not found. Please ensure the model exists in the same directory.")
    exit()

# Load the saved model
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

print(f"Model '{model_filename}' loaded successfully.")


# Predict from console input
def predict_species():
    print("\nEnter Iris flower measurements for prediction:")
    try:
        sepal_length = float(input("Sepal length: "))
        sepal_width = float(input("Sepal width: "))
        petal_length = float(input("Petal length: "))
        petal_width = float(input("Petal width: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    # Prepare input data
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                              columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

    # Predict
    prediction = loaded_model.predict(input_data)
    predicted_species = le.inverse_transform(prediction)
    print(f"Predicted Iris species: {predicted_species[0]}")


# Call the prediction function
if __name__ == "__main__":
    predict_species()
