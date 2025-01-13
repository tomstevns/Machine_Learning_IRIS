# Importér nødvendige biblioteker
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# Generér et syntetisk regressionsdatasæt
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# Split datasæt i træning og test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Opret en polynomisk regressionsmodel (grad=2)
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

# Evaluer modellen på testdatasættet
poly_predictions = poly_model.predict(X_test)
poly_mse = mean_squared_error(y_test, poly_predictions)
print(f"Polynomisk MSE: {poly_mse:.2f}")

# Valgfrit: Visualiser data og model
import matplotlib.pyplot as plt

# Plot træningsdata
plt.scatter(X, y, color='blue', label='Data')

# Plot modellen
import numpy as np
X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_plot = poly_model.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', label='Polynomisk Regression (grad=2)')

plt.title("Polynomisk Regression")
plt.xlabel("Feature")
plt.ylabel("Target")
plt.legend()
plt.show()
