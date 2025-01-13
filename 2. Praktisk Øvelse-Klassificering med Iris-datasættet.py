# Beskrivelse: Vi træner en logistisk regression på Iris-datasættet for at klassificere blomsterarter.
# Løsning (Python-kode):
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
# Indlæs datasæt
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv(url, names=columns)
# Kod arter som tal
data['species'] = data['species'].astype('category').cat.codes
# Split datasæt
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn model
model = LogisticRegression()
model.fit(X_train, y_train)

# Forudsig og evaluer
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
