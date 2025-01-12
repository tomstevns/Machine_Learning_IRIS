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
        Dense(3, activation='softmax')
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

# Evaluate model
loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Accuracy: {accuracy}")

# Predict new data
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Example input
prediction = model.predict(new_sample)
predicted_class = le.inverse_transform([prediction.argmax()])
print(f"Predicted Species: {predicted_class[0]}")
