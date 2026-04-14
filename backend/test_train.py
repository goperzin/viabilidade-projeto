"""Testes para POST /train (T-005)."""

import io
import os
import sys

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

def _make_csv_bytes(n: int = 80, missing_col: str | None = None) -> bytes:
    """Gera um CSV valido (ou com coluna ausente) como bytes."""
    rng = np.random.default_rng(1)
    investment      = rng.uniform(10_000, 200_000, n)
    expected_return = rng.uniform(5_000,  300_000, n)
    impact_score    = rng.uniform(1, 10, n)
    viability       = ((expected_return > investment) & (impact_score > 5)).astype(int)
    df = pd.DataFrame({
        "investment":      investment,
        "expected_return": expected_return,
        "impact_score":    impact_score,
        "viability":       viability,
    })
    if missing_col:
        df = df.drop(columns=[missing_col])
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _post_csv(csv_bytes: bytes, filename: str = "data.csv") -> object:
    """Envia multipart/form-data para POST /train."""
    return client.post(
        "/train",
        files={"file": (filename, io.BytesIO(csv_bytes), "text/csv")},
    )


def _cleanup() -> None:
    for path in (ml.MODEL_PATH, ml.SCALER_PATH, ml.METRICS_PATH):
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Testes — fluxo feliz
# ---------------------------------------------------------------------------

def test_train_returns_200() -> None:
    _cleanup()
    resp = _post_csv(_make_csv_bytes())
    assert resp.status_code == 200, f"Esperado 200, got {resp.status_code}: {resp.text}"
    print("  [OK] POST /train retorna HTTP 200 com CSV valido")


def test_train_response_body_success() -> None:
    data = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes()), "text/csv")},
    ).json()
    assert data["success"] is True, "success deve ser True"
    assert data["message"] == "Modelo treinado com sucesso."
    assert data["metrics"] is not None
    print("  [OK] POST /train retorna success=true e message correta")


def test_train_response_metrics_structure() -> None:
    data = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes()), "text/csv")},
    ).json()
    metrics = data["metrics"]
    assert "accuracy" in metrics, "metrics deve conter 'accuracy'"
    assert "0" in metrics, "metrics deve conter classe '0'"
    assert "1" in metrics, "metrics deve conter classe '1'"
    for cls in ("0", "1"):
        for key in ("precision", "recall", "f1-score"):
            assert key in metrics[cls], f"metrics['{cls}'] deve ter '{key}'"
    print(f"  [OK] POST /train -> metricas com accuracy={metrics['accuracy']:.2f}")


def test_train_persists_artifacts() -> None:
    _cleanup()
    _post_csv(_make_csv_bytes())
    for path in (ml.MODEL_PATH, ml.SCALER_PATH, ml.METRICS_PATH):
        assert os.path.exists(path), f"Artefato nao encontrado: {path}"
    print("  [OK] POST /train persiste os 3 artefatos .joblib")


def test_train_updates_status() -> None:
    _cleanup()
    assert client.get("/status").json()["model_trained"] is False
    _post_csv(_make_csv_bytes())
    assert client.get("/status").json()["model_trained"] is True
    print("  [OK] GET /status reflete model_trained=true apos POST /train")


# ---------------------------------------------------------------------------
# Testes — validacao de arquivo
# ---------------------------------------------------------------------------

def test_train_rejects_non_csv_extension() -> None:
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("data.txt", io.BytesIO(b"col1,col2\n1,2"), "text/plain")},
    )
    assert resp.status_code == 422, f"Esperado 422, got {resp.status_code}"
    assert "csv" in resp.json()["detail"].lower()
    print("  [OK] POST /train rejeita arquivo nao-.csv com 422")


def test_train_rejects_missing_viability_column() -> None:
    _cleanup()
    resp = _post_csv(_make_csv_bytes(missing_col="viability"))
    assert resp.status_code == 422, f"Esperado 422, got {resp.status_code}"
    detail = resp.json()["detail"]
    assert "viability" in detail, f"Mensagem deve citar coluna ausente: {detail}"
    print("  [OK] POST /train rejeita CSV sem coluna 'viability' com 422")


def test_train_rejects_missing_investment_column() -> None:
    _cleanup()
    resp = _post_csv(_make_csv_bytes(missing_col="investment"))
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "investment" in detail
    print("  [OK] POST /train rejeita CSV sem coluna 'investment' com 422")


def test_train_error_message_mentions_missing_columns() -> None:
    _cleanup()
    resp = _post_csv(_make_csv_bytes(missing_col="impact_score"))
    detail = resp.json()["detail"]
    assert "Colunas ausentes no CSV" in detail, f"Mensagem incorreta: {detail}"
    print("  [OK] POST /train usa mensagem padrao 'Colunas ausentes no CSV'")


def test_train_content_type_json() -> None:
    resp = _post_csv(_make_csv_bytes())
    assert "application/json" in resp.headers["content-type"]
    print("  [OK] POST /train retorna Content-Type: application/json")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_train_returns_200,
        test_train_response_body_success,
        test_train_response_metrics_structure,
        test_train_persists_artifacts,
        test_train_updates_status,
        test_train_rejects_non_csv_extension,
        test_train_rejects_missing_viability_column,
        test_train_rejects_missing_investment_column,
        test_train_error_message_mentions_missing_columns,
        test_train_content_type_json,
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
