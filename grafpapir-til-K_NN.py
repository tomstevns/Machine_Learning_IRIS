import matplotlib.pyplot as plt

# Opret tomt grafpapir
plt.figure(figsize=(8, 6))

# Definer aksegrænser og grid
plt.xlim(0, 7)
plt.ylim(0, 3)
plt.xticks(range(0, 8))
plt.yticks(range(0, 4))
plt.grid(color="gray", linestyle="--", linewidth=0.5)

# Tilføj akseetiketter og titel
plt.title("Grafpapir til K-NN Øvelse", fontsize=14)
plt.xlabel("Petal Length", fontsize=12)
plt.ylabel("Petal Width", fontsize=12)

# Tilføj tekstboks til instruktioner
instructions = """
Instruktioner:
1. Plot data for tre Iris-klasser:
   - Setosa (blå cirkler)
   - Versicolor (grønne trekanter)
   - Virginica (orange firkanter)
2. Plot det nye punkt (rød stjerne).
3. Beregn afstande og klassificér med k=3.
"""
#plt.text(7.5, 2.5, instructions, fontsize=10, ha="right", va="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

# Gem grafpapiret som PNG
plt.savefig("grafpapir_knn.png", dpi=300, bbox_inches="tight")
plt.show()
