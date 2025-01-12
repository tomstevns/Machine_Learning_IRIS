# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load the Iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=columns)

# Encode species column
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# Split the dataset
X = iris.iloc[:, :-1]
Y = iris.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the best-performing model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Save the model to the current directory
model_filename = "iris_model.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved as {model_filename}")

# Predict from console input
def predict_species():
    print("\nEnter Iris flower measurements for prediction:")
    sepal_length = float(input("Sepal length: "))
    sepal_width = float(input("Sepal width: "))
    petal_length = float(input("Petal length: "))
    petal_width = float(input("Petal width: "))

    # Load the saved model
    with open(model_filename, 'rb') as file:
        loaded_model = pickle.load(file)

    # Predict
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                               columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
    prediction = loaded_model.predict(input_data)
    predicted_species = le.inverse_transform(prediction)
    print(f"Predicted Iris species: {predicted_species[0]}")

# Call the prediction function
if __name__ == "__main__":
    predict_species()
