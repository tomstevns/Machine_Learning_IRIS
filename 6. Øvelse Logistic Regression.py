#       *********Logistic Regression********
#
# er en simpel metode til at lave klassifikation,
# altså at bestemme, hvilken kategori noget hører til.
# I stedet for at finde en lige linje (som i lineær regression),
# forsøger Logistic Regression at finde en "S-formet" kurve,
# der kan adskille data i forskellige grupper.

# I denne applikation bruges Logistic Regression til at forudsige,
# hvilken af de tre typer Iris-blomster (Setosa, Versicolor, Virginica)
# en blomst sandsynligvis er, baseret på målinger som længde og bredde
# af blomstens blade (sepal og petal).

# Det fungerer ved at beregne sandsynligheden for, at en prøve tilhører en bestemt gruppe,
# og vælger derefter den gruppe med den højeste sandsynlighed.


#           ****** Classification Report  ******
#
# er en oversigt over, hvor godt en klassifikationsmodel præsterer.
# Det viser tre vigtige metrikker for hver kategori:

# 1. Precision: Hvor mange af de forudsigelser, der blev lavet for en kategori,
#    der faktisk var korrekte. Det svarer til, hvor "præcist" modellen rammer rigtigt.

# 2. Recall: Hvor mange af de faktiske prøver fra en kategori, som modellen korrekt fandt.
#    Det måler, hvor god modellen er til at opdage alt, der tilhører en kategori.

# 3. F1-score: En kombination af precision og recall. Det er et gennemsnit,
#    der giver et samlet billede af, hvordan modellen præsterer for en kategori.

# 4. Support: Antallet af prøver i hver kategori. Det viser, hvor mange data modellen blev testet på.

# Eksempel:
# Hvis vi har tre typer Iris-blomster (Setosa, Versicolor, Virginica),
# viser rapporten, hvordan modellen klarer sig for hver af disse typer.

#    ***** Forskellen mellem Precision og Recall:  ******

# Precision:
# Precision handler om, hvor præcise vores forudsigelser er.
# Det fortæller os, hvor mange af de ting, modellen har forudsagt som positive,
# der faktisk er positive.
# Tænk på det som "hvor mange af de pletter, vi skød på, der ramte målet".

# Formel:
# Precision = (Antal rigtige positive forudsigelser) / (Alle positive forudsigelser)
# Eksempel: Hvis vi forudsagde 10 ting som "blomster af en bestemt type",
# men kun 7 af dem var korrekte, er precisionen 7/10 = 70%.

# Recall:
# Recall handler om, hvor godt modellen er til at finde ALLE de rigtige ting.
# Det måler, hvor mange af de faktiske positive ting modellen korrekt har opdaget.
# Tænk på det som "hvor mange af alle mål på banen, vi ramte".

# Formel:
# Recall = (Antal rigtige positive forudsigelser) / (Alle faktiske positive)
# Eksempel: Hvis der er 20 blomster af en bestemt type, og vi korrekt fandt 15 af dem,
# er recall 15/20 = 75%.

# Sammenligning:
# - Precision er vigtig, når vi vil være sikre på, at det, vi forudsiger som positivt,
#   faktisk er korrekt (undgå falske alarmer).
# - Recall er vigtig, når vi vil sikre, at vi finder ALLE positive ting,
#   selv hvis vi laver nogle fejl (undgå at overse noget vigtigt).

# Eksempel:
# Hvis vi forudsiger, om en blomst er af typen "Setosa":
# - Høj precision betyder, at næsten alle, vi siger er "Setosa", faktisk er det.
# - Høj recall betyder, at vi fanger næsten alle "Setosa", selv hvis vi fejlagtigt
#   inkluderer nogle, der ikke er "Setosa".


# Importér nødvendige biblioteker
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Indlæs Iris-datasættet
iris = load_iris()
X = iris.data  # Funktioner: sepal og petal målinger
y = iris.target  # Målvariabel: blomsterarter
target_names = iris.target_names  # Navne på blomsterarter

# Split datasæt i træning og test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Træn Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Forudsig testdatasættet
predictions = model.predict(X_test)

# Evaluer nøjagtighed
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion matrix
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=target_names))
