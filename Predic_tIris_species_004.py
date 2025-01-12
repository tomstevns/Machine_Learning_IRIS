
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Define the Flask app
app = Flask(__name__)

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
    raise FileNotFoundError(f"Model file '{model_filename}' not found. Please ensure the model exists in the same directory.")

# Load the saved model
with open(model_filename, 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve user input from form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare input data
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                   columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])

        # Predict
        prediction = loaded_model.predict(input_data)
        predicted_species = le.inverse_transform(prediction)[0]

        return render_template('index.html',
                               prediction_text=f'Predicted Iris species: {predicted_species}')
    except Exception as e:
        return render_template('index.html',
                               prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
