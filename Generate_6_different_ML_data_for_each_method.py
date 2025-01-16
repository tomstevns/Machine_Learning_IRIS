import pandas as pd
import numpy as np

# Logistic Regression Dataset (Finansielle data)
np.random.seed(42)
logistic_data = {
    "Alder": np.random.randint(18, 70, size=150),
    "Indkomst": np.random.randint(20000, 150000, size=150),
    "Kredit_score": np.random.randint(300, 850, size=150),
    "Antal_\u00e5bne_konti": np.random.randint(1, 10, size=150),
    "G\u00e6ldsandel": np.round(np.random.uniform(0.1, 0.9, size=150), 2),
    "Godkendt": np.random.choice([0, 1], size=150)
}
logistic_df = pd.DataFrame(logistic_data)

# Linear Discriminant Analysis Dataset (Medicinsk diagnose)
lda_data = {
    "Blodtryk": np.random.randint(90, 180, size=150),
    "Kolesterol": np.random.randint(150, 300, size=150),
    "BMI": np.round(np.random.uniform(18.5, 40.0, size=150), 1),
    "Alder": np.random.randint(18, 90, size=150),
    "Fysisk_aktivitetsniveau": np.random.randint(1, 5, size=150),
    "Risiko": np.random.choice(["Lav", "Medium", "H\u00f8j"], size=150)
}
lda_df = pd.DataFrame(lda_data)

# K-Nearest Neighbors Dataset (Bioinformatik)
knn_data = {
    "Frekvens_A": np.random.randint(100, 500, size=150),
    "Frekvens_T": np.random.randint(100, 500, size=150),
    "Frekvens_C": np.random.randint(100, 500, size=150),
    "Frekvens_G": np.random.randint(100, 500, size=150),
    "Sekvensl\u00e6ngde": np.random.randint(1000, 5000, size=150),
    "Gene_type": np.random.choice(["Husgen", "Regulerende", "Strukturgener"], size=150)
}
knn_df = pd.DataFrame(knn_data)

# Na\u00efve Bayes Dataset (E-mail spamfiltrering)
nb_data = {
    "Ord_Gratis": np.random.poisson(2, size=150),
    "Ord_Tilbud": np.random.poisson(3, size=150),
    "Ord_Betaling": np.random.poisson(1, size=150),
    "Ord_Konto": np.random.poisson(4, size=150),
    "Spam": np.random.choice(["Spam", "Ikke-spam"], size=150)
}
nb_df = pd.DataFrame(nb_data)

# Decision Tree Dataset (Produktionsoptimering)
cart_data = {
    "Materialekvalitet": np.random.randint(1, 10, size=150),
    "Arbejdstimer": np.random.randint(5, 50, size=150),
    "Maskintype": np.random.choice(["Type_A", "Type_B", "Type_C"], size=150),
    "Omkostninger_per_enhed": np.random.randint(100, 1000, size=150),
    "Produktionsvalg": np.random.choice(["A", "B", "C"], size=150)
}
cart_df = pd.DataFrame(cart_data)

# Support Vector Machine Dataset (Billedklassifikation)
svm_data = {
    "Gennemsnitlig_pixelintensitet": np.random.uniform(0.1, 1.0, size=150),
    "Kanter": np.random.uniform(0.0, 1.0, size=150),
    "Teksturparametre": np.random.uniform(0.1, 0.9, size=150),
    "Objekttype": np.random.choice(["Cirkel", "Trekant", "Firkant"], size=150)
}
svm_df = pd.DataFrame(svm_data)

# Save datasets to CSV (uncomment to save locally)
logistic_df.to_csv("logistic_regression_data.csv", index=False)
lda_df.to_csv("linear_discriminant_analysis_data.csv", index=False)
knn_df.to_csv("k_nearest_neighbors_data.csv", index=False)
nb_df.to_csv("naive_bayes_data.csv", index=False)
cart_df.to_csv("decision_tree_data.csv", index=False)
svm_df.to_csv("support_vector_machine_data.csv", index=False)
