from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import pandas as pd
import os

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, names=columns)

# Encode target variable
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# Split dataset
X = iris.iloc[:, :-1].values
Y = to_categorical(iris.iloc[:, -1].values)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Check if model already exists
model_filename = "iris_model.h5"
if os.path.exists(model_filename):
    # Load the existing model
    model = load_model(model_filename)
    print("Model indlæst fra iris_model.h5")
else:
    # Build and train a new model
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # 3 output classes for Iris dataset
    ])

    # Compile model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    model.fit(X_train, Y_train, epochs=50, batch_size=8, validation_data=(X_test, Y_test))

    # Save the model
    model.save(model_filename)
    print("Model gemt som iris_model.h5")

# Function to predict new data based on console input
def predict_from_console():
    print("\nIndtast værdier for en Iris-blomst for at lave en forudsigelse.")
    try:
        sepal_length = float(input("Sepal længde: "))
        sepal_width = float(input("Sepal bredde: "))
        petal_length = float(input("Petal længde: "))
        petal_width = float(input("Petal bredde: "))
    except ValueError:
        print("Ugyldig input. Sørg for at indtaste numeriske værdier.")
        return

    # Prepare input data
    new_sample = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(new_sample)
    predicted_class = le.inverse_transform([prediction.argmax()])
    print(f"Forudsagt Iris-art: {predicted_class[0]}")

# Call the prediction function
if __name__ == "__main__":
    predict_from_console()
