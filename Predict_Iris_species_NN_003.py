from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import pandas as pd
import os

# Flask app initialization
app = Flask(__name__)

# Load dataset for encoding species
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, names=columns)

# Encode target variable
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# Load trained model
model_filename = "iris_model.h5"
if os.path.exists(model_filename):
    model = load_model(model_filename)
    print("Model indlæst fra iris_model.h5")
else:
    raise FileNotFoundError("Modellen iris_model.h5 findes ikke. Sørg for at træne og gemme modellen først.")


# Flask routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Create input for prediction
        new_sample = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Make prediction
        prediction = model.predict(new_sample)
        predicted_class = le.inverse_transform([prediction.argmax()])
        return render_template('index.html', prediction_text=f"Forudsagt Iris-art: {predicted_class[0]}")
    except ValueError:
        return render_template('index.html',
                               prediction_text="Ugyldig input! Sørg for at indtaste gyldige numeriske værdier.")


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
