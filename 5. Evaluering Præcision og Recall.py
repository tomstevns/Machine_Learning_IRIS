# Importér nødvendige biblioteker
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Indlæs Iris-datasættet
iris = load_iris()
X = iris.data  # Funktioner (features): sepal/petal længder og bredder
y = iris.target  # Målvariabel (target): blomsterarter
target_names = iris.target_names  # Navne på blomsterarter

# Split datasættet i trænings- og testdatasæt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn k-NN model
knn = KNeighborsClassifier(n_neighbors=3)  # k=3 betyder tre nærmeste naboer
knn.fit(X_train, y_train)

# Forudsig på testdatasættet
knn_predictions = knn.predict(X_test)

# Generer classification report
print(classification_report(y_test, knn_predictions, target_names=target_names))
