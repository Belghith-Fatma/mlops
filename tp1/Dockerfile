
# 1. Image de base : Linux minimal avec Python 3.11
FROM python:3.11-slim

# 2. Dossier de travail dans l'image
WORKDIR /app

# 3. Copier et installer les dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copier le code et le modèle
COPY app.py .
COPY model/ model/

# 5. Charger le modèle depuis le dossier local
ENV MODEL_URI=model

# 6. Port utilisé par l'API
EXPOSE 8000

# 7. Démarrer FastAPI avec Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
