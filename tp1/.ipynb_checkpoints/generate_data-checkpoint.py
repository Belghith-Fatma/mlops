import pandas as pd
import numpy as np

# Fixer la graine aléatoire pour garantir la reproductibilité
np.random.seed(42)

# Nombre de clients fictifs
n = 500

# Création des données
data = pd.DataFrame({
    "age": np.random.randint(18, 70, n),
    "salaire": np.random.randint(1200, 8000, n),
    "abonnement": np.random.choice(
        ["Basique", "Standard", "Premium"],
        n
    ),
    "anciennete": np.random.randint(0, 15, n),
    "consommation": np.random.randint(0, 100, n)
})

# Création de la variable cible : churn
# churn = 1 si :
# - consommation < 30
# - anciennete < 2 ans
# Sinon churn = 0
data["churn"] = (
    (data["consommation"] < 30)
    & (data["anciennete"] < 2)
).astype(int)

# Sauvegarder les données dans un fichier CSV
data.to_csv("churn_data.csv", index=False)

# Afficher les 5 premières lignes
print(data.head())

# Afficher le taux de churn
print(f"Taux de churn : {data['churn'].mean():.1%}")