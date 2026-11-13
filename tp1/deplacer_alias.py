import mlflow
from mlflow import MlflowClient

# Connexion a la base MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

# 1. Retrouver le run "essai-2"
runs = mlflow.search_runs(
    experiment_names=["churn-prediction"],
    filter_string="tags.mlflow.runName = 'essai-2'"
)

if runs.empty:
    raise ValueError("Aucun run nomme 'essai-2' trouve.")

run_id = runs.iloc[0]["run_id"]

print("Run trouve :", run_id)

# 2. Enregistrer le modele : creation d'une nouvelle version
v = mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="churn-model"
)

print("Nouvelle version creee :", v.version)

# 3. Recuperer la version actuellement designee par champion
actuel = client.get_model_version_by_alias(
    name="churn-model",
    alias="champion"
)

ancienne_version = actuel.version

print("Champion actuel : version", ancienne_version)

# 4. Deplacer temporairement l'alias champion vers la nouvelle version
client.set_registered_model_alias(
    name="churn-model",
    alias="champion",
    version=v.version
)

nouveau_champion = client.get_model_version_by_alias(
    name="churn-model",
    alias="champion"
)

print(
    "Apres deplacement : champion -> version",
    nouveau_champion.version
)

# 5. Remettre l'alias champion sur la version precedente
client.set_registered_model_alias(
    name="churn-model",
    alias="champion",
    version=ancienne_version
)

champion_final = client.get_model_version_by_alias(
    name="churn-model",
    alias="champion"
)

print(
    "Apres retour : champion -> version",
    champion_final.version
)