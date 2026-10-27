from typing import Literal

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Creer l'application
app = FastAPI(title="API Churn")

# 2. Charger le modele champion une seule fois
mlflow.set_tracking_uri("sqlite:///mlflow.db")

model = mlflow.sklearn.load_model(
    "models:/churn-model@champion"
)

# Recuperer les colonnes attendues par le modele
features = list(model.feature_names_in_)

# 3. Definir les donnees attendues
class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

# 4. Route pour verifier que l'API fonctionne
@app.get("/health")
def health():
    return {"status": "ok"}

# 5. Route pour predire le churn
@app.post("/predict")
def predict(client: Client):
    # Transformer les donnees du client en DataFrame
    ligne = pd.DataFrame([client.model_dump()])

    # Transformer abonnement en colonnes numeriques
    ligne = pd.get_dummies(ligne, columns=["abonnement"], dtype=int)

    # Garder les colonnes attendues par le modele
    ligne = ligne.reindex(columns=features, fill_value=0)

    # Effectuer la prediction
    resultat = int(model.predict(ligne)[0])

    # Retourner le resultat au format JSON
    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }