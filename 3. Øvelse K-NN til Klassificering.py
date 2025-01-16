import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import load_iris

# Indlæs Iris-datasættet
iris = load_iris()
X = iris.data[:, 2:4]  # Brug kun 'petal_length' og 'petal_width'
y = iris.target

# Split data i trænings- og testdatasæt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tag input fra brugeren
k = int(input("Indtast værdien for k (fx 3): "))
petal_length = float(input("Indtast Petal Length (fx 4.8): "))
petal_width = float(input("Indtast Petal Width (fx 1.6): "))
new_point = np.array([[petal_length, petal_width]])

# Træn K-NN modellen
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Forudsig med testdata og det nye punkt
y_pred = knn.predict(X_test)
new_prediction = knn.predict(new_point)

# Vis resultat for det nye punkt
print(f"\nFor det nye punkt ({petal_length}, {petal_width}):")
print(f"Den forudsagte klasse er: {iris.target_names[new_prediction[0]]}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
