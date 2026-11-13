
import os
from typing import Literal

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --------------------------------------------------
# 1. Initialisation de FastAPI
# --------------------------------------------------

app = FastAPI(
    title="API Churn",
    description="API de prédiction du départ des clients",
    version="1.0.0"
)

# --------------------------------------------------
# 2. Chargement du modèle local
# --------------------------------------------------

MODEL_PATH = os.getenv("MODEL_PATH", "model")

model = mlflow.sklearn.load_model(MODEL_PATH)

print(f"Modèle chargé depuis : {MODEL_PATH}")

# --------------------------------------------------
# 3. Définition des données d'entrée
# --------------------------------------------------

class Client(BaseModel):
    age: int
    salaire: float
    abonnement: Literal["Basique", "Standard", "Premium"]
    anciennete: float
    consommation: float

# --------------------------------------------------
# 4. Route d'accueil
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Bienvenue dans l'API Churn",
        "documentation": "/docs",
        "health": "/health"
    }

# --------------------------------------------------
# 5. Vérification de l'API
# --------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# --------------------------------------------------
# 6. Prédiction du churn
# --------------------------------------------------

@app.post("/predict")
def predict(client: Client):

    ligne = pd.DataFrame([client.model_dump()])

    ligne = pd.get_dummies(ligne)

    ligne = ligne.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    resultat = int(model.predict(ligne)[0])

    return {
        "churn": resultat,
        "message": (
            "Le client va partir"
            if resultat == 1
            else "Le client reste"
        )
    }
