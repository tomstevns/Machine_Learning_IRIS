# Importér nødvendige biblioteker
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Indlæs Iris-datasættet
iris = load_iris()
X = iris.data  # Funktioner (features): sepal/petal længder og bredder
y = iris.target  # Målvariabel (target): blomsterarter

# Split datasættet i trænings- og testdatasæt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn k-NN model med k=3
knn = KNeighborsClassifier(n_neighbors=3)  # k=3 betyder tre nærmeste naboer
knn.fit(X_train, y_train)

# Forudsig på testdatasættet
knn_predictions = knn.predict(X_test)

# Evaluer modellens nøjagtighed
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"K-NN Accuracy (k=3): {knn_accuracy:.2f}")

# Udskriv forudsigelser og faktiske værdier (valgfrit)
print(f"Test Predictions: {knn_predictions}")
print(f"True Labels:      {y_test}")
