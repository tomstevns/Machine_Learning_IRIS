# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, header=None, names=columns)

# Explore the dataset
print(iris.head())
print(iris.describe())
print(iris['species'].value_counts())

# Visualize the dataset
sns.pairplot(iris, hue='species')
plt.show()

# Encode species column
le = LabelEncoder()
iris['species'] = le.fit_transform(iris['species'])

# Split the dataset
X = iris.iloc[:, :-1]
Y = iris.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Spot Check Algorithms (Fix for Cell 18/19)
models = [
    ('LR', LogisticRegression()),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC())
]

# Set up KFold with shuffle enabled
seed = 42
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

# Evaluate each model
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Train and evaluate the best-performing model
best_model = LogisticRegression()  # Example: Replace with best model after evaluation
best_model.fit(X_train, Y_train)
predictions = best_model.predict(X_test)

# Classification report
print("Classification Report:")
print(classification_report(Y_test, predictions))

# Confusion matrix
cm = confusion_matrix(Y_test, predictions)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
