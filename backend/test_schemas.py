"""Testes de unidade para schemas.py (T-003)."""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from pydantic import ValidationError

from schemas import ProjectInput, PredictionResponse, TrainResponse, StatusResponse


# ---------------------------------------------------------------------------
# ProjectInput
# ---------------------------------------------------------------------------

def test_project_input_valid() -> None:
    p = ProjectInput(investment=40000, expected_return=60000, impact_score=6)
    assert p.investment      == 40000
    assert p.expected_return == 60000
    assert p.impact_score    == 6
    print("  [OK] ProjectInput aceita dados validos")


def test_project_input_float_coercion() -> None:
    p = ProjectInput(investment="10000", expected_return="20000.5", impact_score="7")
    assert isinstance(p.investment, float)
    assert isinstance(p.expected_return, float)
    assert isinstance(p.impact_score, float)
    print("  [OK] ProjectInput converte strings numericas para float")


def test_project_input_missing_field() -> None:
    try:
        ProjectInput(investment=10000, expected_return=20000)
        assert False, "Deveria ter lancado ValidationError"
    except ValidationError:
        pass
    print("  [OK] ProjectInput rejeita campo obrigatorio ausente")


def test_project_input_invalid_type() -> None:
    try:
        ProjectInput(investment="nao-numero", expected_return=20000, impact_score=5)
        assert False, "Deveria ter lancado ValidationError"
    except ValidationError:
        pass
    print("  [OK] ProjectInput rejeita valor nao-numerico")


# ---------------------------------------------------------------------------
# PredictionResponse
# ---------------------------------------------------------------------------

def test_prediction_response_viable() -> None:
    r = PredictionResponse(viability=1, viability_label="Viavel", probability=0.87)
    assert r.viability       == 1
    assert r.viability_label == "Viavel"
    assert r.probability     == 0.87
    print("  [OK] PredictionResponse aceita projeto viavel")


def test_prediction_response_inviable() -> None:
    r = PredictionResponse(viability=0, viability_label="Inviavel", probability=0.23)
    assert r.viability       == 0
    assert r.viability_label == "Inviavel"
    assert r.probability     == 0.23
    print("  [OK] PredictionResponse aceita projeto inviavel")


def test_prediction_response_missing_field() -> None:
    try:
        PredictionResponse(viability=1, probability=0.9)
        assert False, "Deveria ter lancado ValidationError"
    except ValidationError:
        pass
    print("  [OK] PredictionResponse rejeita campo obrigatorio ausente")


# ---------------------------------------------------------------------------
# TrainResponse
# ---------------------------------------------------------------------------

def test_train_response_success() -> None:
    metrics = {"accuracy": 0.9, "0": {}, "1": {}}
    r = TrainResponse(success=True, message="Modelo treinado com sucesso.", metrics=metrics)
    assert r.success is True
    assert r.metrics == metrics
    print("  [OK] TrainResponse aceita resposta de sucesso com metricas")


def test_train_response_failure() -> None:
    r = TrainResponse(success=False, message="Colunas ausentes: ['viability']", metrics=None)
    assert r.success  is False
    assert r.metrics  is None
    print("  [OK] TrainResponse aceita resposta de falha com metrics=None")


def test_train_response_metrics_default_none() -> None:
    r = TrainResponse(success=True, message="ok")
    assert r.metrics is None
    print("  [OK] TrainResponse.metrics tem default None")


# ---------------------------------------------------------------------------
# StatusResponse
# ---------------------------------------------------------------------------

def test_status_response_trained() -> None:
    metrics = {"accuracy": 0.89, "0": {}, "1": {}}
    r = StatusResponse(model_trained=True, metrics=metrics)
    assert r.model_trained is True
    assert r.metrics       == metrics
    print("  [OK] StatusResponse aceita modelo treinado com metricas")


def test_status_response_not_trained() -> None:
    r = StatusResponse(model_trained=False, metrics=None)
    assert r.model_trained is False
    assert r.metrics       is None
    print("  [OK] StatusResponse aceita modelo nao treinado com metrics=None")


def test_status_response_metrics_default_none() -> None:
    r = StatusResponse(model_trained=False)
    assert r.metrics is None
    print("  [OK] StatusResponse.metrics tem default None")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_project_input_valid,
        test_project_input_float_coercion,
        test_project_input_missing_field,
        test_project_input_invalid_type,
        test_prediction_response_viable,
        test_prediction_response_inviable,
        test_prediction_response_missing_field,
        test_train_response_success,
        test_train_response_failure,
        test_train_response_metrics_default_none,
        test_status_response_trained,
        test_status_response_not_trained,
        test_status_response_metrics_default_none,
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

    print(f"\n{'='*50}")
    print(f"Resultado: {passed} passou(aram) | {failed} falhou(aram)")
    if failed:
        sys.exit(1)
