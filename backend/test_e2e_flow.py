"""Teste do fluxo completo end-to-end (T-011).

Simula exatamente a sequência de chamadas que app.js executa:
  checkStatus() → handleUpload() → checkStatus() → handlePredict() → renderResult()

Valida não só os status HTTP, mas que as payloads têm a estrutura exata
que renderMetrics() e renderResult() esperam consumir.
"""

import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from fastapi.testclient import TestClient

import model as ml
from main import app

client = TestClient(app)

# CSV de exemplo gerado na raiz do projeto
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "projects_data.csv")

# Dados do formulário especificados no SPRINT-001 (T-011)
PREDICT_PAYLOAD = {
    "investment":      40_000,
    "expected_return": 60_000,
    "impact_score":    6,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cleanup() -> None:
    for path in (ml.MODEL_PATH, ml.SCALER_PATH, ml.METRICS_PATH):
        if os.path.exists(path):
            os.remove(path)


def _csv_bytes_from_file() -> bytes:
    """Lê o projects_data.csv da raiz do projeto."""
    with open(CSV_PATH, "rb") as f:
        return f.read()


def _csv_bytes_minimal(n: int = 80) -> bytes:
    """Gera um CSV sintético como fallback se o arquivo não existir."""
    import numpy as np
    rng = np.random.default_rng(42)
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


def _get_csv_bytes() -> bytes:
    if os.path.exists(CSV_PATH):
        return _csv_bytes_from_file()
    return _csv_bytes_minimal()


# ---------------------------------------------------------------------------
# Passo 1 — checkStatus() sem modelo treinado
# ---------------------------------------------------------------------------

def test_passo1_status_inicial_sem_modelo() -> None:
    """checkStatus(): modelo ausente → upload-section visível, predict oculto."""
    _cleanup()
    resp = client.get("/status")

    assert resp.status_code == 200, f"GET /status retornou {resp.status_code}"
    data = resp.json()

    # Campos que app.js acessa em checkStatus()
    assert "model_trained" in data, "Campo model_trained ausente na resposta"
    assert "metrics" in data,       "Campo metrics ausente na resposta"

    assert data["model_trained"] is False, "model_trained deve ser false antes do treino"
    assert data["metrics"] is None,        "metrics deve ser null antes do treino"


# ---------------------------------------------------------------------------
# Passo 2 — handleUpload() com projects_data.csv
# ---------------------------------------------------------------------------

def test_passo2_upload_csv_valido() -> None:
    """handleUpload(): envia projects_data.csv → success: true, metrics presentes."""
    csv_bytes = _get_csv_bytes()

    resp = client.post(
        "/train",
        files={"file": ("projects_data.csv", io.BytesIO(csv_bytes), "text/csv")},
    )

    assert resp.status_code == 200, f"POST /train retornou {resp.status_code}: {resp.text}"
    data = resp.json()

    # Campos que handleUpload() verifica: data.success e data.message
    assert data["success"] is True,         "success deve ser true"
    assert isinstance(data["message"], str), "message deve ser string"
    assert len(data["message"]) > 0,         "message não deve ser vazio"

    # Métricas presentes (exibidas após treino)
    assert data["metrics"] is not None, "metrics deve ser retornado após treino"
    assert "accuracy" in data["metrics"], "accuracy ausente nas métricas"
    assert 0.0 <= data["metrics"]["accuracy"] <= 1.0, "accuracy deve estar entre 0 e 1"


# ---------------------------------------------------------------------------
# Passo 3 — checkStatus() após treino
# ---------------------------------------------------------------------------

def test_passo3_status_apos_treino() -> None:
    """checkStatus(): modelo presente → metrics-section e predict-section visíveis."""
    resp = client.get("/status")

    assert resp.status_code == 200
    data = resp.json()

    assert data["model_trained"] is True, "model_trained deve ser true após treino"
    assert data["metrics"] is not None,   "metrics deve estar presente após treino"

    metrics = data["metrics"]

    # Estrutura exata que renderMetrics() acessa
    assert "accuracy" in metrics,          "metrics.accuracy ausente"
    assert "1" in metrics,                 "metrics['1'] ausente (classe positiva)"
    assert "precision"  in metrics["1"],   "metrics['1'].precision ausente"
    assert "recall"     in metrics["1"],   "metrics['1'].recall ausente"
    assert "f1-score"   in metrics["1"],   "metrics['1']['f1-score'] ausente"

    # Valores plausíveis para renderMetrics() formatar como percentual
    for key in ("precision", "recall", "f1-score"):
        val = metrics["1"][key]
        assert 0.0 <= val <= 1.0, f"metrics['1']['{key}'] deve estar entre 0 e 1, got {val}"


# ---------------------------------------------------------------------------
# Passo 4 — handlePredict() com dados do SPRINT-001
# ---------------------------------------------------------------------------

def test_passo4_predict_com_dados_do_sprint() -> None:
    """handlePredict(): envia investment=40000, expected_return=60000, impact_score=6."""
    resp = client.post("/predict", json=PREDICT_PAYLOAD)

    assert resp.status_code == 200, f"POST /predict retornou {resp.status_code}: {resp.text}"
    data = resp.json()

    # Campos que renderResult() acessa
    assert "viability"       in data, "Campo viability ausente"
    assert "viability_label" in data, "Campo viability_label ausente"
    assert "probability"     in data, "Campo probability ausente"

    # renderResult(): data.viability === 1 ou === 0
    assert data["viability"] in (0, 1), f"viability deve ser 0 ou 1, got {data['viability']}"

    # badge.textContent = data.viability_label
    assert isinstance(data["viability_label"], str),  "viability_label deve ser string"
    assert len(data["viability_label"]) > 0,           "viability_label não deve ser vazio"

    # fill.style.width = `${Math.round(data.probability * 100)}%`
    assert 0.0 <= data["probability"] <= 1.0, \
        f"probability deve estar entre 0 e 1, got {data['probability']}"


# ---------------------------------------------------------------------------
# Passo 5 — renderResult(): consistência entre viability e viability_label
# ---------------------------------------------------------------------------

def test_passo5_viability_label_consistente_com_viability() -> None:
    """Verifica que viability_label bate com viability (0 → Inviável, 1 → Viável)."""
    resp = client.post("/predict", json=PREDICT_PAYLOAD)
    data = resp.json()

    viability = data["viability"]
    label     = data["viability_label"]

    if viability == 1:
        assert "viável" in label.lower() or "viavel" in label.lower(), \
            f"viability=1 mas label='{label}'"
    else:
        assert "inviável" in label.lower() or "inviavel" in label.lower(), \
            f"viability=0 mas label='{label}'"


# ---------------------------------------------------------------------------
# Passo 6 — barra de progresso: probability pode ser formatada como percentual
# ---------------------------------------------------------------------------

def test_passo6_probability_formatavel_como_percentual() -> None:
    """probability * 100 deve resultar em inteiro entre 0 e 100 (para progress bar)."""
    resp = client.post("/predict", json=PREDICT_PAYLOAD)
    data = resp.json()

    pct = round(data["probability"] * 100)
    assert 0 <= pct <= 100, f"Percentual fora do intervalo: {pct}%"


# ---------------------------------------------------------------------------
# Passo 7 — projects_data.csv tem pelo menos 50 linhas (SPRINT-001 nota)
# ---------------------------------------------------------------------------

def test_passo7_csv_tem_linhas_suficientes() -> None:
    """O CSV de exemplo deve ter ao menos 50 linhas de dados (nota do SPRINT-001)."""
    assert os.path.exists(CSV_PATH), \
        f"projects_data.csv não encontrado em {CSV_PATH}"

    df = pd.read_csv(CSV_PATH)
    assert len(df) >= 50, f"CSV deve ter ao menos 50 linhas, tem {len(df)}"

    required = {"investment", "expected_return", "impact_score", "viability"}
    missing  = required - set(df.columns)
    assert not missing, f"Colunas ausentes no CSV: {missing}"


# ---------------------------------------------------------------------------
# Runner manual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    steps = [
        ("Passo 1 — status sem modelo",          test_passo1_status_inicial_sem_modelo),
        ("Passo 2 — upload projects_data.csv",   test_passo2_upload_csv_valido),
        ("Passo 3 — status após treino",          test_passo3_status_apos_treino),
        ("Passo 4 — predict (dados do sprint)",  test_passo4_predict_com_dados_do_sprint),
        ("Passo 5 — label consistente",           test_passo5_viability_label_consistente_com_viability),
        ("Passo 6 — probability → percentual",   test_passo6_probability_formatavel_como_percentual),
        ("Passo 7 — CSV tem ≥ 50 linhas",        test_passo7_csv_tem_linhas_suficientes),
    ]

    passed = failed = 0
    for label, fn in steps:
        try:
            fn()
            print(f"  [OK] {label}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {label}: {exc}")
            failed += 1

    _cleanup()
    print(f"\nResultado: {passed}/{passed+failed} passou(aram)")
    if failed:
        sys.exit(1)
