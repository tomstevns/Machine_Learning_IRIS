# Importér nødvendige biblioteker
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generér et syntetisk regressionsdatasæt
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# Split datasæt i træning og test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn en lineær regressionsmodel
model = LinearRegression()
model.fit(X_train, y_train)

# Forudsig testdatasættet
predictions = model.predict(X_test)

# Evaluer model med Mean Squared Error
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse:.2f}")

# Visualisér residualer
plt.scatter(y_test, predictions - y_test, color='blue', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Sande værdier")
plt.ylabel("Residualer")
plt.title("Residualanalyse")
plt.show()
