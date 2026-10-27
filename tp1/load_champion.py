import mlflow
import mlflow.sklearn
import pandas as pd

# Connexion a la base MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. Charger le modele champion
model = mlflow.sklearn.load_model("models:/churn-model@champion")

print("Modele charge :", type(model).__name__)
print("Variables attendues :", list(model.feature_names_in_))

# 2. Creer un nouveau client
client = pd.DataFrame([{
    "age": 35,
    "salaire": 2500,
    "anciennete": 1,
    "consommation": 10,
    "abonnement_Premium": False,
    "abonnement_Standard": True
}])

# 3. Verifier les variables attendues
features = list(model.feature_names_in_)

missing = [col for col in features if col not in client.columns]
if missing:
    raise ValueError(f"Variables manquantes : {missing}")

# Garder les colonnes dans le bon ordre
client = client[features]

print("\nDonnees du client :")
print(client)

# 4. Predire le depart du client
prediction = model.predict(client)[0]

print("\nLe client va-t-il partir ? ->", "OUI" if prediction == 1 else "NON")

# 5. Afficher les probabilites si disponibles
if hasattr(model, "predict_proba"):
    probabilites = model.predict_proba(client)[0]
    print("\nProbabilites :")
    for classe, probabilite in zip(model.classes_, probabilites):
        print(f"Classe {classe} : {probabilite:.2%}")