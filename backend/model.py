"""Lógica de ML: treino, previsão e métricas do modelo de viabilidade de projetos."""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Constantes de path
# ---------------------------------------------------------------------------

ARTIFACTS_DIR = "artifacts"
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "logistic_model.joblib")
SCALER_PATH   = os.path.join(ARTIFACTS_DIR, "logistic_model_scaler.joblib")
METRICS_PATH  = os.path.join(ARTIFACTS_DIR, "logistic_model_metrics.joblib")

REQUIRED_COLUMNS = {"investment", "expected_return", "impact_score", "viability"}

FEATURES = ["investment", "expected_return", "impact_score"]
TARGET   = "viability"


# ---------------------------------------------------------------------------
# Funções públicas
# ---------------------------------------------------------------------------

def model_exists() -> bool:
    """Retorna True se os três artefatos do modelo existem em disco."""
    return all(os.path.exists(p) for p in (MODEL_PATH, SCALER_PATH, METRICS_PATH))


def train_model(df: pd.DataFrame) -> dict:
    """Treina o modelo de regressão logística, persiste os artefatos e retorna as métricas.

    Args:
        df: DataFrame com as colunas obrigatórias já validadas.

    Returns:
        Dicionário de métricas no formato do classification_report do sklearn.
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    report: dict = classification_report(y_test, y_pred, output_dict=True)

    # Manter apenas as chaves relevantes para a resposta da API
    metrics = {
        "0":        report.get("0", report.get(0, {})),
        "1":        report.get("1", report.get(1, {})),
        "accuracy": report.get("accuracy", 0.0),
    }

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model,   MODEL_PATH)
    joblib.dump(scaler,  SCALER_PATH)
    joblib.dump(metrics, METRICS_PATH)

    return metrics


def predict(project: dict) -> dict:
    """Carrega o modelo salvo e retorna a previsão de viabilidade para um projeto.

    Args:
        project: Dicionário com as chaves 'investment', 'expected_return', 'impact_score'.

    Returns:
        Dicionário com 'viability' (int), 'viability_label' (str) e 'probability' (float).
    """
    model: LogisticRegression = joblib.load(MODEL_PATH)
    scaler: StandardScaler    = joblib.load(SCALER_PATH)

    features = np.array([[
        project["investment"],
        project["expected_return"],
        project["impact_score"],
    ]])

    features_scaled = scaler.transform(features)

    viability: int        = int(model.predict(features_scaled)[0])
    probability: float    = float(model.predict_proba(features_scaled)[0][viability])
    viability_label: str  = "Viável" if viability == 1 else "Inviável"

    return {
        "viability":       viability,
        "viability_label": viability_label,
        "probability":     round(probability, 4),
    }


def get_metrics() -> dict | None:
    """Retorna as métricas salvas em disco, ou None se o modelo não foi treinado."""
    if not os.path.exists(METRICS_PATH):
        return None
    return joblib.load(METRICS_PATH)
