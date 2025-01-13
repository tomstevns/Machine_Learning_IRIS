import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Indlæs Boston Housing-datasæt fra ekstern kilde
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
data = pd.read_csv(url)

# Funktioner og mål
X = data.drop(columns=['medv'])  # 'medv' er målvariablen (medianværdi af ejendomme)
y = data['medv']

# Split datasæt
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn lineær regression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Evaluer model
predictions = reg.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse:.2f}")
