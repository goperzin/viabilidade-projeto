"""Testes de integracao e CORS (T-007).

Cobre:
- Cabecalhos CORS em todos os endpoints
- Preflight OPTIONS
- Fluxo completo: GET /status -> POST /train -> GET /status -> POST /predict
"""

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

ORIGIN = "http://localhost:3000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv_bytes(n: int = 80) -> bytes:
    rng = np.random.default_rng(3)
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
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _cleanup() -> None:
    for path in (ml.MODEL_PATH, ml.SCALER_PATH, ml.METRICS_PATH):
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Testes de CORS
# ---------------------------------------------------------------------------

def test_cors_header_on_status() -> None:
    resp = client.get("/status", headers={"Origin": ORIGIN})
    assert "access-control-allow-origin" in resp.headers, \
        "GET /status deve retornar header Access-Control-Allow-Origin"
    assert resp.headers["access-control-allow-origin"] == "*"
    print("  [OK] GET /status retorna Access-Control-Allow-Origin: *")


def test_cors_header_on_train() -> None:
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes()), "text/csv")},
        headers={"Origin": ORIGIN},
    )
    assert "access-control-allow-origin" in resp.headers
    assert resp.headers["access-control-allow-origin"] == "*"
    print("  [OK] POST /train retorna Access-Control-Allow-Origin: *")


def test_cors_header_on_predict() -> None:
    resp = client.post(
        "/predict",
        json={"investment": 40000, "expected_return": 60000, "impact_score": 6},
        headers={"Origin": ORIGIN},
    )
    # 200 (modelo treinado) ou 400 (sem modelo) — ambos devem ter CORS
    assert "access-control-allow-origin" in resp.headers
    assert resp.headers["access-control-allow-origin"] == "*"
    print(f"  [OK] POST /predict retorna Access-Control-Allow-Origin: * (status={resp.status_code})")


def test_cors_preflight_options_status() -> None:
    resp = client.options(
        "/status",
        headers={
            "Origin":                        ORIGIN,
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code in (200, 204), \
        f"OPTIONS /status deve retornar 200 ou 204, got {resp.status_code}"
    print(f"  [OK] OPTIONS /status retorna {resp.status_code} (preflight aceito)")


def test_cors_preflight_options_train() -> None:
    resp = client.options(
        "/train",
        headers={
            "Origin":                         ORIGIN,
            "Access-Control-Request-Method":  "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code in (200, 204)
    print(f"  [OK] OPTIONS /train retorna {resp.status_code} (preflight aceito)")


def test_cors_preflight_options_predict() -> None:
    resp = client.options(
        "/predict",
        headers={
            "Origin":                         ORIGIN,
            "Access-Control-Request-Method":  "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert resp.status_code in (200, 204)
    print(f"  [OK] OPTIONS /predict retorna {resp.status_code} (preflight aceito)")


# ---------------------------------------------------------------------------
# Teste de fluxo completo (end-to-end via TestClient)
# ---------------------------------------------------------------------------

def test_full_flow_end_to_end() -> None:
    """Fluxo: status (sem modelo) -> train -> status (com modelo) -> predict."""
    _cleanup()

    # 1. Status inicial: modelo nao treinado
    data = client.get("/status").json()
    assert data["model_trained"] is False
    assert data["metrics"] is None
    print("  [1/4] GET /status -> model_trained=false OK")

    # 2. Treinar com CSV valido
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes()), "text/csv")},
    )
    assert resp.status_code == 200
    train_data = resp.json()
    assert train_data["success"] is True
    assert train_data["metrics"] is not None
    print(f"  [2/4] POST /train -> success=true, accuracy={train_data['metrics']['accuracy']:.2f} OK")

    # 3. Status atualizado: modelo treinado
    data = client.get("/status").json()
    assert data["model_trained"] is True
    assert data["metrics"] is not None
    assert abs(data["metrics"]["accuracy"] - train_data["metrics"]["accuracy"]) < 0.001
    print("  [3/4] GET /status -> model_trained=true, metricas consistentes OK")

    # 4. Prever com dados validos
    resp = client.post(
        "/predict",
        json={"investment": 40000, "expected_return": 60000, "impact_score": 6},
    )
    assert resp.status_code == 200
    pred = resp.json()
    assert pred["viability"] in (0, 1)
    assert pred["viability_label"] in ("Viavel", "Inviavel", "Viável", "Inviável")
    assert 0.0 <= pred["probability"] <= 1.0
    print(f"  [4/4] POST /predict -> {pred['viability_label']} ({pred['probability']:.2%}) OK")

    print("  [OK] Fluxo completo executado com sucesso")


# ---------------------------------------------------------------------------
# Teste de todos os endpoints respondendo (smoke test)
# ---------------------------------------------------------------------------

def test_all_endpoints_reachable() -> None:
    endpoints = [
        ("GET",  "/status",  None),
        ("POST", "/train",   None),   # sem arquivo -> 422, mas endpoint existe
        ("POST", "/predict", None),   # sem body -> 422, mas endpoint existe
    ]
    for method, path, _ in endpoints:
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path)
        # Qualquer status exceto 404/405 indica que o endpoint existe e responde
        assert resp.status_code != 404, f"{method} {path} retornou 404"
        assert resp.status_code != 405, f"{method} {path} retornou 405 (Method Not Allowed)"
    print("  [OK] Todos os endpoints respondem (nao retornam 404/405)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_cors_header_on_status,
        test_cors_header_on_train,
        test_cors_header_on_predict,
        test_cors_preflight_options_status,
        test_cors_preflight_options_train,
        test_cors_preflight_options_predict,
        test_full_flow_end_to_end,
        test_all_endpoints_reachable,
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
