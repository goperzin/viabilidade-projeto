"""Testes de unidade para model.py (T-002)."""

import os
import sys
import numpy as np
import pandas as pd

# Garante que o módulo é importado a partir do diretório do backend
sys.path.insert(0, os.path.dirname(__file__))

# Altera o cwd para backend/ para que os paths relativos dos artefatos funcionem
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import model  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 80) -> pd.DataFrame:
    """Gera DataFrame sintético balanceado para treino."""
    rng = np.random.default_rng(0)
    investment       = rng.uniform(10_000, 200_000, n)
    expected_return  = rng.uniform(5_000,  300_000, n)
    impact_score     = rng.uniform(1, 10, n)
    # Rótulo simples: viável se retorno > investimento e impacto > 5
    viability = ((expected_return > investment) & (impact_score > 5)).astype(int)
    return pd.DataFrame({
        "investment":      investment,
        "expected_return": expected_return,
        "impact_score":    impact_score,
        "viability":       viability,
    })


def _cleanup() -> None:
    """Remove artefatos gerados durante os testes."""
    for path in (model.MODEL_PATH, model.SCALER_PATH, model.METRICS_PATH):
        if os.path.exists(path):
            os.remove(path)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_model_not_exists_before_train() -> None:
    _cleanup()
    assert model.model_exists() is False, "model_exists deve ser False antes do treino"
    print("  [OK] model_exists() -> False antes do treino")


def test_get_metrics_none_before_train() -> None:
    _cleanup()
    assert model.get_metrics() is None, "get_metrics deve retornar None antes do treino"
    print("  [OK] get_metrics() -> None antes do treino")


def test_train_model_returns_metrics() -> None:
    _cleanup()
    df      = _make_df()
    metrics = model.train_model(df)

    assert isinstance(metrics, dict), "train_model deve retornar dict"
    assert "accuracy" in metrics,    "métricas devem conter 'accuracy'"
    assert "0" in metrics,           "métricas devem conter classe '0'"
    assert "1" in metrics,           "métricas devem conter classe '1'"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "accuracy deve estar entre 0 e 1"

    for cls in ("0", "1"):
        for key in ("precision", "recall", "f1-score"):
            assert key in metrics[cls], f"métricas['{cls}'] deve ter '{key}'"

    print(f"  [OK] train_model() -> accuracy={metrics['accuracy']:.2f}")


def test_artifacts_persisted_after_train() -> None:
    for path in (model.MODEL_PATH, model.SCALER_PATH, model.METRICS_PATH):
        assert os.path.exists(path), f"Artefato não encontrado: {path}"
    print("  [OK] Artefatos .joblib persistidos em disco")


def test_model_exists_after_train() -> None:
    assert model.model_exists() is True, "model_exists deve ser True após o treino"
    print("  [OK] model_exists() -> True após o treino")


def test_get_metrics_after_train() -> None:
    metrics = model.get_metrics()
    assert metrics is not None, "get_metrics deve retornar dict após o treino"
    assert "accuracy" in metrics
    print("  [OK] get_metrics() -> dict com métricas após o treino")


def test_predict_viable_project() -> None:
    result = model.predict({
        "investment":      40_000,
        "expected_return": 60_000,
        "impact_score":    8,
    })
    assert "viability"       in result, "resultado deve conter 'viability'"
    assert "viability_label" in result, "resultado deve conter 'viability_label'"
    assert "probability"     in result, "resultado deve conter 'probability'"
    assert result["viability"] in (0, 1), "viability deve ser 0 ou 1"
    assert result["viability_label"] in ("Viável", "Inviável")
    assert 0.0 <= result["probability"] <= 1.0, "probability deve estar entre 0 e 1"
    print(f"  [OK] predict() -> {result['viability_label']} ({result['probability']:.2%})")


def test_predict_inviable_project() -> None:
    result = model.predict({
        "investment":      100_000,
        "expected_return": 10_000,
        "impact_score":    2,
    })
    assert result["viability"] in (0, 1)
    assert result["viability_label"] in ("Viável", "Inviável")
    assert 0.0 <= result["probability"] <= 1.0
    print(f"  [OK] predict() inviável -> {result['viability_label']} ({result['probability']:.2%})")


def test_probability_rounds_to_4_decimals() -> None:
    result = model.predict({
        "investment":      50_000,
        "expected_return": 80_000,
        "impact_score":    7,
    })
    # Verifica que a probability tem no máximo 4 casas decimais
    decimals = len(str(result["probability"]).split(".")[-1])
    assert decimals <= 4, f"probability deve ter no máximo 4 casas decimais, got {decimals}"
    print(f"  [OK] probability arredondada para 4 casas: {result['probability']}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_model_not_exists_before_train,
        test_get_metrics_none_before_train,
        test_train_model_returns_metrics,
        test_artifacts_persisted_after_train,
        test_model_exists_after_train,
        test_get_metrics_after_train,
        test_predict_viable_project,
        test_predict_inviable_project,
        test_probability_rounds_to_4_decimals,
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
        finally:
            pass  # não limpa entre testes — alguns dependem do estado anterior

    _cleanup()

    print(f"\n{'='*50}")
    print(f"Resultado: {passed} passou(aram) | {failed} falhou(aram)")
    if failed:
        sys.exit(1)
