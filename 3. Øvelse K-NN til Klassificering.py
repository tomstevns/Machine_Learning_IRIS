#    ******  K-NN (k-Nearest Neighbors)   *******
#
#er en simpel metode til klassifikation og regression.
# Den fungerer ved at kigge på de nærmeste naboer i dataene for at lave en forudsigelse.

# Hvordan det virker:
# - Når vi skal forudsige en ny prøve (f.eks. hvilken type blomst det er),
#   finder K-NN de "k" nærmeste naboer (andre prøver i datasættet) baseret på afstand.
# - Afstanden måles normalt med Euclideansk afstand (en lige linje mellem punkterne).
# - Modellen ser på naboernes klasser og vælger den mest almindelige (majoritet) som forudsigelse.

# K:
# - "k" er antallet af naboer, modellen kigger på.
# - F.eks. hvis k=3, ser modellen på de 3 nærmeste naboer for at træffe en beslutning.
# - Et lavt k (f.eks. 1) kan føre til en model, der overtilpasser (overfitting).
# - Et højt k (f.eks. 10) kan føre til en model, der generaliserer for meget (underfitting).

# Fordele:
# - Let at forstå og implementere.
# - Virker godt, når dataene har en klar struktur.

# Ulemper:
# - K-NN kan være langsom, når der er mange data, fordi den skal beregne afstand til alle naboer.
# - Sensitiv over for støj i data.

# I denne applikation:
# K-NN bruges til at klassificere Iris-blomster. Når vi vil forudsige, hvilken type blomst
# en ny prøve er, ser modellen på målingerne (sepal og petal længde/bredde)
# og sammenligner dem med målingerne for andre blomster i datasættet.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

# Indlæs Iris-datasættet
iris = load_iris()
X = iris.data[:, 2:4]  # Brug kun 'petal_length' og 'petal_width' for 2D-visualisering
y = iris.target

# Nyt punkt til klassifikation
new_point = np.array([[4.8, 1.6]])

# Funktion til at visualisere K-NN for forskellige værdier af k
def plot_knn_multiple_k(k_values, X, y, new_point):
    fig, axes = plt.subplots(1, len(k_values), figsize=(20, 5), sharey=True)
    colors = ['blue', 'green', 'orange']  # Farver for klasser
    for idx, k in enumerate(k_values):
        # Træn K-NN med k naboer
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)

        # Forudsig beslutningsgrænser
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot beslutningsgrænser og træningsdata
        axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
        for i, color in enumerate(colors):
            axes[idx].scatter(X[y == i, 0], X[y == i, 1], c=color, label=iris.target_names[i])
        axes[idx].scatter(new_point[0][0], new_point[0][1], c='red', edgecolor='black', s=100, label="Nyt Punkt")
        axes[idx].set_title(f"K = {k}")
        axes[idx].set_xlabel("Petal Length")
        if idx == 0:
            axes[idx].set_ylabel("Petal Width")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()

# Visualiser for k = 1, 3, 5, 7, 9
plot_knn_multiple_k(k_values=[1, 3, 5, 7, 9], X=X, y=y, new_point=new_point)
