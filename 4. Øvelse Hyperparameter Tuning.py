# Importér nødvendige biblioteker
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Indlæs Iris-datasættet
iris = load_iris()
X = iris.data  # Funktioner (features): sepal/petal længder og bredder
y = iris.target  # Målvariabel (target): blomsterarter

# Split datasættet i trænings- og testdatasæt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definér parameternetværk for k-NN
param_grid = {'n_neighbors': [3, 5, 7, 9]}

# Opsæt GridSearchCV
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Udskriv bedste parametre og score
print(f"Bedste parametre: {grid.best_params_}")
print(f"Bedste score (cross-validering): {grid.best_score_:.2f}")

# Evaluer den bedste model på testdatasættet
best_knn = grid.best_estimator_  # Den bedste model fundet af GridSearchCV
test_predictions = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"Nøjagtighed på testdata: {test_accuracy:.2f}")
