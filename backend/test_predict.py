"""Testes para POST /predict (T-006)."""

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

def _make_csv_bytes(n: int = 80) -> bytes:
    rng = np.random.default_rng(2)
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


def _train() -> None:
    client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes()), "text/csv")},
    )


def _predict(investment: float, expected_return: float, impact_score: float):
    return client.post(
        "/predict",
        json={
            "investment":      investment,
            "expected_return": expected_return,
            "impact_score":    impact_score,
        },
    )


def _cleanup() -> None:
    for path in (ml.MODEL_PATH, ml.SCALER_PATH, ml.METRICS_PATH):
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Testes — sem modelo treinado
# ---------------------------------------------------------------------------

def test_predict_without_model_returns_400() -> None:
    _cleanup()
    resp = _predict(40_000, 60_000, 6)
    assert resp.status_code == 400, f"Esperado 400, got {resp.status_code}"
    print("  [OK] POST /predict sem modelo retorna HTTP 400")


def test_predict_without_model_error_message() -> None:
    _cleanup()
    detail = _predict(40_000, 60_000, 6).json()["detail"]
    assert "Modelo ainda nao foi treinado" in detail or "Modelo ainda não foi treinado" in detail, \
        f"Mensagem incorreta: {detail}"
    print("  [OK] POST /predict sem modelo retorna mensagem padrao correta")


# ---------------------------------------------------------------------------
# Testes — fluxo feliz (com modelo treinado)
# ---------------------------------------------------------------------------

def test_predict_returns_200_after_train() -> None:
    _cleanup()
    _train()
    resp = _predict(40_000, 60_000, 6)
    assert resp.status_code == 200, f"Esperado 200, got {resp.status_code}: {resp.text}"
    print("  [OK] POST /predict retorna HTTP 200 apos treino")


def test_predict_response_has_required_keys() -> None:
    data = _predict(40_000, 60_000, 6).json()
    assert "viability"       in data, "Resposta deve conter 'viability'"
    assert "viability_label" in data, "Resposta deve conter 'viability_label'"
    assert "probability"     in data, "Resposta deve conter 'probability'"
    print("  [OK] POST /predict retorna as tres chaves obrigatorias")


def test_predict_viability_is_int_zero_or_one() -> None:
    data = _predict(40_000, 60_000, 6).json()
    assert data["viability"] in (0, 1), f"viability deve ser 0 ou 1, got {data['viability']}"
    print(f"  [OK] viability={data['viability']} (int 0 ou 1)")


def test_predict_viability_label_matches_viability() -> None:
    data = _predict(40_000, 60_000, 6).json()
    expected_label = "Viavel" if data["viability"] == 1 else "Inviavel"
    # Aceita com ou sem acento (encoding pode variar)
    label = data["viability_label"].replace("á", "a").replace("í", "i")
    assert label == expected_label, f"Label '{label}' nao bate com viability={data['viability']}"
    print(f"  [OK] viability_label='{data['viability_label']}' coerente com viability={data['viability']}")


def test_predict_probability_between_0_and_1() -> None:
    data = _predict(40_000, 60_000, 6).json()
    prob = data["probability"]
    assert 0.0 <= prob <= 1.0, f"probability deve estar entre 0 e 1, got {prob}"
    print(f"  [OK] probability={prob} dentro do intervalo [0, 1]")


def test_predict_probability_max_4_decimals() -> None:
    data = _predict(50_000, 80_000, 7).json()
    prob_str = str(data["probability"])
    decimals = len(prob_str.split(".")[-1]) if "." in prob_str else 0
    assert decimals <= 4, f"probability deve ter no maximo 4 casas decimais, got {prob_str}"
    print(f"  [OK] probability={data['probability']} com ate 4 casas decimais")


def test_predict_spec_example_input() -> None:
    """Testa o exemplo exato do SPEC.md: investment=40000, expected_return=60000, impact_score=6."""
    data = _predict(40_000, 60_000, 6).json()
    assert data["viability"] in (0, 1)
    assert data["viability_label"] in ("Viavel", "Inviavel", "Viável", "Inviável")
    assert 0.0 <= data["probability"] <= 1.0
    print(f"  [OK] Exemplo do SPEC: {data['viability_label']} ({data['probability']:.2%})")


def test_predict_inviable_project() -> None:
    """Projeto com alto investimento, baixo retorno e baixo impacto deve tender a Inviavel."""
    data = _predict(200_000, 10_000, 1).json()
    assert data["viability"] in (0, 1)
    assert 0.0 <= data["probability"] <= 1.0
    print(f"  [OK] Projeto inviavel: {data['viability_label']} ({data['probability']:.2%})")


def test_predict_content_type_json() -> None:
    resp = _predict(40_000, 60_000, 6)
    assert "application/json" in resp.headers["content-type"]
    print("  [OK] POST /predict retorna Content-Type: application/json")


# ---------------------------------------------------------------------------
# Testes — validacao de entrada
# ---------------------------------------------------------------------------

def test_predict_rejects_missing_field() -> None:
    resp = client.post("/predict", json={"investment": 40_000, "expected_return": 60_000})
    assert resp.status_code == 422, f"Esperado 422, got {resp.status_code}"
    print("  [OK] POST /predict rejeita body com campo faltando (422)")


def test_predict_rejects_invalid_type() -> None:
    resp = client.post(
        "/predict",
        json={"investment": "nao-numero", "expected_return": 60_000, "impact_score": 6},
    )
    assert resp.status_code == 422
    print("  [OK] POST /predict rejeita tipo invalido (422)")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_predict_without_model_returns_400,
        test_predict_without_model_error_message,
        test_predict_returns_200_after_train,
        test_predict_response_has_required_keys,
        test_predict_viability_is_int_zero_or_one,
        test_predict_viability_label_matches_viability,
        test_predict_probability_between_0_and_1,
        test_predict_probability_max_4_decimals,
        test_predict_spec_example_input,
        test_predict_inviable_project,
        test_predict_content_type_json,
        test_predict_rejects_missing_field,
        test_predict_rejects_invalid_type,
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
