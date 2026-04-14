"""Testes para GET /status (T-004)."""

import os
import sys

# Garante imports a partir do backend/
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import model as ml
from main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    investment      = rng.uniform(10_000, 200_000, n)
    expected_return = rng.uniform(5_000,  300_000, n)
    impact_score    = rng.uniform(1, 10, n)
    viability       = ((expected_return > investment) & (impact_score > 5)).astype(int)
    return pd.DataFrame({
        "investment":      investment,
        "expected_return": expected_return,
        "impact_score":    impact_score,
        "viability":       viability,
    })


def _cleanup() -> None:
    for path in (ml.MODEL_PATH, ml.SCALER_PATH, ml.METRICS_PATH):
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_status_returns_200() -> None:
    _cleanup()
    resp = client.get("/status")
    assert resp.status_code == 200, f"Esperado 200, got {resp.status_code}"
    print("  [OK] GET /status retorna HTTP 200")


def test_status_not_trained_body() -> None:
    _cleanup()
    data = client.get("/status").json()
    assert data["model_trained"] is False, "model_trained deve ser False sem modelo"
    assert data["metrics"] is None, "metrics deve ser null sem modelo"
    print("  [OK] GET /status -> model_trained=false, metrics=null quando nao treinado")


def test_status_has_required_keys() -> None:
    _cleanup()
    data = client.get("/status").json()
    assert "model_trained" in data, "Resposta deve conter 'model_trained'"
    assert "metrics" in data, "Resposta deve conter 'metrics'"
    print("  [OK] GET /status retorna as chaves obrigatorias")


def test_status_trained_body() -> None:
    _cleanup()
    ml.train_model(_make_df())
    data = client.get("/status").json()
    assert data["model_trained"] is True, "model_trained deve ser True apos treino"
    assert data["metrics"] is not None, "metrics nao deve ser null apos treino"
    assert "accuracy" in data["metrics"], "metrics deve conter 'accuracy'"
    assert "0" in data["metrics"], "metrics deve conter classe '0'"
    assert "1" in data["metrics"], "metrics deve conter classe '1'"
    print(f"  [OK] GET /status -> model_trained=true, accuracy={data['metrics']['accuracy']:.2f}")


def test_status_metrics_values_are_floats() -> None:
    data = client.get("/status").json()
    assert isinstance(data["metrics"]["accuracy"], float), "accuracy deve ser float"
    for cls in ("0", "1"):
        for key in ("precision", "recall", "f1-score"):
            val = data["metrics"][cls][key]
            assert isinstance(val, float), f"metrics['{cls}']['{key}'] deve ser float"
    print("  [OK] GET /status -> todos os valores de metricas sao float")


def test_status_content_type_json() -> None:
    resp = client.get("/status")
    assert "application/json" in resp.headers["content-type"]
    print("  [OK] GET /status retorna Content-Type: application/json")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_status_returns_200,
        test_status_not_trained_body,
        test_status_has_required_keys,
        test_status_trained_body,
        test_status_metrics_values_are_floats,
        test_status_content_type_json,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            print(f"\n{test.__name__}")
            test()
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failed += 1

    _cleanup()
    print(f"\n{'='*50}")
    print(f"Resultado: {passed} passou(aram) | {failed} falhou(aram)")
    if failed:
        sys.exit(1)
