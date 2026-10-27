
# ============================================================
# 1. Importer les bibliotheques
# ============================================================
import mlflow
from mlflow import MlflowClient

# ============================================================
# 2. Se connecter a la base MLflow du TP3
# ============================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# ============================================================
# 3. Retrouver le meilleur run selon le score F1
# ============================================================
experiment_name = "churn-prediction"

experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    raise ValueError(
        f"Experience '{experiment_name}' introuvable. "
        "Verifiez le fichier mlflow.db."
    )

# Recuperer les runs termines, classes par F1 decroissant
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.f1 DESC"],
    max_results=1000
)

# Verifier que des runs existent
if runs.empty:
    raise ValueError("Aucun run termine trouve.")

# Eliminer les runs sans score F1
runs = runs.dropna(subset=["metrics.f1"])

if runs.empty:
    raise ValueError(
        "Aucun run ne contient la metrique 'f1'. "
        "Verifiez le nom de la metrique."
    )

# Selectionner le meilleur run
best = runs.iloc[0]
run_id = best["run_id"]
f1_score = best["metrics.f1"]
run_name = best.get("tags.mlflow.runName", "Run sans nom")

print("Meilleur run :", run_name)
print("Score F1     :", f1_score)
print("Run ID       :", run_id[:8], "...")

# ============================================================
# 4. Enregistrer le modele dans le Model Registry
# ============================================================
model_uri = f"runs:/{run_id}/model"
model_name = "churn-model"

# Verifier que le modele peut etre charge depuis le run
try:
    mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    raise RuntimeError(
        f"Impossible de charger le modele depuis {model_uri}. "
        "Verifiez que le modele a bien ete enregistre sous "
        "le chemin 'model' dans ce run."
    ) from e

# Enregistrer le modele (creation d'une nouvelle version)
result = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print(
    f"Modele enregistre : {model_name} "
    f"| version {result.version}"
)

# ============================================================
# 5. Definir l'alias champion
# ============================================================
client = MlflowClient()

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=result.version
)

print(
    f"Alias 'champion' -> version {result.version}"
)

# ============================================================
# 6. Verifier le resultat
# ============================================================
champion = client.get_model_version_by_alias(
    name=model_name,
    alias="champion"
)

print("\nVerification du Registry :")
print("Nom du modele :", champion.name)
print("Version       :", champion.version)

# Charger le modele champion pour une utilisation ulterieure
model = mlflow.pyfunc.load_model(
    f"models:/{model_name}@champion"
)

print("Modele champion charge avec succes.")
