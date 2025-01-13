# Importér nødvendige biblioteker
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Generér et syntetisk regressionsdatasæt
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)

# Split datasæt i træning og test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge(alpha=1.0)  # Alpha er reguleringsstyrken
ridge.fit(X_train, y_train)
ridge_predictions = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print(f"Ridge Regression MSE: {ridge_mse:.2f}")

# Lasso Regression
lasso = Lasso(alpha=0.1)  # Alpha er reguleringsstyrken
lasso.fit(X_train, y_train)
lasso_predictions = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_predictions)
print(f"Lasso Regression MSE: {lasso_mse:.2f}")
