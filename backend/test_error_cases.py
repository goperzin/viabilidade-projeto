"""Testes dos casos de erro (T-012).

Cobre os quatro cenários descritos no SPRINT-001:
  1. Upload de arquivo não-CSV → mensagem de erro
  2. Upload de CSV sem coluna obrigatória → mensagem de erro
  3. Campos do formulário em branco / inválidos → validação client-side (HTML attrs)
  4. POST /predict sem modelo treinado → mensagem de aviso padrão

Além disso, cobre cenários não testados anteriormente:
  - CSV com conteúdo não parseável (binário)
  - CSV com colunas corretas mas zero linhas de dados
  - Upload sem campo `file`
  - POST /predict com body vazio ou campos extras
  - Respostas de erro não expõem stack traces
  - Mensagens de erro seguem o padrão do CLAUDE.md
"""

import io
import os
import sys
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import model as ml
from main import app

client = TestClient(app)

HTML_PATH = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv_bytes(n: int = 80, missing_col: str | None = None) -> bytes:
    rng = np.random.default_rng(99)
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


def _cleanup() -> None:
    for path in (ml.MODEL_PATH, ml.SCALER_PATH, ml.METRICS_PATH):
        if os.path.exists(path):
            os.remove(path)


def _train() -> None:
    client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes()), "text/csv")},
    )


class _HTMLAttrs(HTMLParser):
    """Coleta atributos por ID."""

    def __init__(self):
        super().__init__()
        self.by_id: dict[str, dict] = {}

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if "id" in d:
            self.by_id[d["id"]] = d


def _parse_html_attrs() -> dict[str, dict]:
    inspector = _HTMLAttrs()
    with open(HTML_PATH, encoding="utf-8") as f:
        inspector.feed(f.read())
    return inspector.by_id


# ===========================================================================
# CENÁRIO 1 — Upload de arquivo não-CSV
# ===========================================================================

def test_upload_txt_retorna_422() -> None:
    """Arquivo .txt deve ser rejeitado com HTTP 422."""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("relatorio.txt", io.BytesIO(b"col1,col2\n1,2"), "text/plain")},
    )
    assert resp.status_code == 422, f"Esperado 422, got {resp.status_code}"


def test_upload_txt_mensagem_padrao() -> None:
    """Mensagem deve seguir o padrão do CLAUDE.md: 'Apenas arquivos .csv são aceitos.'"""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("data.txt", io.BytesIO(b"a,b\n1,2"), "text/plain")},
    )
    detail = resp.json()["detail"]
    assert "csv" in detail.lower(), f"Mensagem deve mencionar 'csv': {detail}"


def test_upload_pdf_retorna_422() -> None:
    """Extensão .pdf também deve ser rejeitada."""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("relatorio.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert resp.status_code == 422


def test_upload_sem_arquivo_retorna_422() -> None:
    """POST /train sem campo 'file' deve retornar 422."""
    _cleanup()
    resp = client.post("/train")
    assert resp.status_code == 422, f"Esperado 422, got {resp.status_code}"


# ===========================================================================
# CENÁRIO 2 — CSV sem coluna obrigatória
# ===========================================================================

def test_csv_sem_viability_retorna_422() -> None:
    """CSV sem coluna 'viability' deve retornar 422."""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes(missing_col="viability")), "text/csv")},
    )
    assert resp.status_code == 422


def test_csv_sem_viability_cita_coluna_ausente() -> None:
    """Mensagem de erro deve citar 'viability' como coluna ausente."""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes(missing_col="viability")), "text/csv")},
    )
    detail = resp.json()["detail"]
    assert "viability" in detail, f"Mensagem deve citar coluna ausente: {detail}"


def test_csv_mensagem_usa_padrao_colunas_ausentes() -> None:
    """Mensagem deve começar com 'Colunas ausentes no CSV' (padrão do CLAUDE.md)."""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(_make_csv_bytes(missing_col="impact_score")), "text/csv")},
    )
    detail = resp.json()["detail"]
    assert "Colunas ausentes no CSV" in detail, f"Mensagem incorreta: {detail}"


def test_csv_binario_retorna_500() -> None:
    """CSV com conteúdo binário não parseável deve retornar 500 com mensagem genérica."""
    _cleanup()
    binario = bytes(range(256)) * 4  # dados binários inválidos
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(binario), "text/csv")},
    )
    assert resp.status_code == 500, f"CSV binário deve retornar 500, got {resp.status_code}"
    detail = resp.json()["detail"]
    assert "Erro interno" in detail, f"Mensagem de erro interno incorreta: {detail}"


def test_csv_vazio_retorna_erro() -> None:
    """CSV com cabeçalho correto mas zero linhas de dados deve ser rejeitado."""
    _cleanup()
    csv_vazio = b"investment,expected_return,impact_score,viability\n"
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(csv_vazio), "text/csv")},
    )
    # Deve retornar 422 (sem dados para treinar) ou 500 (falha no treino)
    assert resp.status_code in (422, 500), \
        f"CSV vazio deve retornar 422 ou 500, got {resp.status_code}"


# ===========================================================================
# CENÁRIO 3 — Campos do formulário em branco (validação client-side — HTML)
# ===========================================================================

def test_campo_investment_tem_required() -> None:
    """Campo #investment deve ter atributo 'required' para validação client-side."""
    attrs = _parse_html_attrs()
    assert "investment" in attrs, "Campo #investment não encontrado no HTML"
    assert "required" in attrs["investment"], "Campo #investment deve ter 'required'"


def test_campo_expected_return_tem_required() -> None:
    """Campo #expected_return deve ter atributo 'required'."""
    attrs = _parse_html_attrs()
    assert "expected_return" in attrs
    assert "required" in attrs["expected_return"]


def test_campo_impact_score_tem_required() -> None:
    """Campo #impact_score deve ter atributo 'required'."""
    attrs = _parse_html_attrs()
    assert "impact_score" in attrs
    assert "required" in attrs["impact_score"]


def test_campo_impact_score_tem_min_e_max() -> None:
    """Campo #impact_score deve ter min=0 e max=10 para evitar valores fora do range."""
    attrs = _parse_html_attrs()
    assert "impact_score" in attrs
    ia = attrs["impact_score"]
    assert "min" in ia, "Campo #impact_score deve ter atributo 'min'"
    assert "max" in ia, "Campo #impact_score deve ter atributo 'max'"
    assert ia["min"] == "0",  f"min deve ser '0', got '{ia['min']}'"
    assert ia["max"] == "10", f"max deve ser '10', got '{ia['max']}'"


def test_campos_numericos_tem_type_number() -> None:
    """Campos do formulário devem ter type='number' para rejeitar texto."""
    attrs = _parse_html_attrs()
    for field_id in ("investment", "expected_return", "impact_score"):
        assert attrs.get(field_id, {}).get("type") == "number", \
            f"Campo #{field_id} deve ter type='number'"


def test_csv_input_tem_required() -> None:
    """Input de arquivo CSV deve ter 'required' para impedir envio vazio."""
    attrs = _parse_html_attrs()
    assert "csv-file" in attrs
    assert "required" in attrs["csv-file"], "Input #csv-file deve ter 'required'"


def test_csv_input_aceita_apenas_csv() -> None:
    """Input de arquivo deve ter accept='.csv' para filtrar no seletor de arquivos."""
    attrs = _parse_html_attrs()
    assert "csv-file" in attrs
    accept = attrs["csv-file"].get("accept", "")
    assert ".csv" in accept, f"Input #csv-file deve ter accept='.csv', got '{accept}'"


# ===========================================================================
# CENÁRIO 4 — POST /predict sem modelo treinado
# ===========================================================================

def test_predict_sem_modelo_retorna_400() -> None:
    """POST /predict sem modelo deve retornar HTTP 400."""
    _cleanup()
    resp = client.post(
        "/predict",
        json={"investment": 40_000, "expected_return": 60_000, "impact_score": 6},
    )
    assert resp.status_code == 400, f"Esperado 400, got {resp.status_code}"


def test_predict_sem_modelo_mensagem_padrao() -> None:
    """Mensagem deve ser exatamente a definida no CLAUDE.md."""
    _cleanup()
    detail = client.post(
        "/predict",
        json={"investment": 40_000, "expected_return": 60_000, "impact_score": 6},
    ).json()["detail"]

    esperada = "Modelo ainda não foi treinado. Faça upload de um CSV primeiro."
    assert detail == esperada, \
        f"Mensagem incorreta.\nEsperada: {esperada}\nRecebida: {detail}"


def test_predict_sem_modelo_exibe_no_campo_detail() -> None:
    """Erro de modelo não treinado deve estar no campo 'detail' (padrão FastAPI HTTPException)."""
    _cleanup()
    data = client.post(
        "/predict",
        json={"investment": 40_000, "expected_return": 60_000, "impact_score": 6},
    ).json()
    assert "detail" in data, "Resposta de erro deve ter campo 'detail'"


# ===========================================================================
# REGRA GERAL — respostas de erro não expõem stack traces
# ===========================================================================

def test_erro_422_nao_expoe_traceback() -> None:
    """Resposta 422 não deve conter 'Traceback' ou 'File \"' (stack trace Python)."""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("data.txt", io.BytesIO(b"a,b\n1,2"), "text/plain")},
    )
    body = resp.text
    assert "Traceback" not in body, "Resposta 422 expõe Traceback Python"
    assert 'File "' not in body,    "Resposta 422 expõe caminho de arquivo do stack trace"


def test_erro_400_nao_expoe_traceback() -> None:
    """Resposta 400 não deve expor stack trace."""
    _cleanup()
    resp = client.post(
        "/predict",
        json={"investment": 40_000, "expected_return": 60_000, "impact_score": 6},
    )
    body = resp.text
    assert "Traceback" not in body
    assert 'File "' not in body


def test_erro_500_nao_expoe_traceback() -> None:
    """Resposta 500 de CSV binário não deve expor stack trace."""
    _cleanup()
    resp = client.post(
        "/train",
        files={"file": ("data.csv", io.BytesIO(bytes(range(256)) * 4), "text/csv")},
    )
    if resp.status_code == 500:
        body = resp.text
        assert "Traceback" not in body, "Resposta 500 expõe Traceback Python"
        assert 'File "' not in body


# ---------------------------------------------------------------------------
# Runner manual
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Cenário 1
        test_upload_txt_retorna_422,
        test_upload_txt_mensagem_padrao,
        test_upload_pdf_retorna_422,
        test_upload_sem_arquivo_retorna_422,
        # Cenário 2
        test_csv_sem_viability_retorna_422,
        test_csv_sem_viability_cita_coluna_ausente,
        test_csv_mensagem_usa_padrao_colunas_ausentes,
        test_csv_binario_retorna_500,
        test_csv_vazio_retorna_erro,
        # Cenário 3
        test_campo_investment_tem_required,
        test_campo_expected_return_tem_required,
        test_campo_impact_score_tem_required,
        test_campo_impact_score_tem_min_e_max,
        test_campos_numericos_tem_type_number,
        test_csv_input_tem_required,
        test_csv_input_aceita_apenas_csv,
        # Cenário 4
        test_predict_sem_modelo_retorna_400,
        test_predict_sem_modelo_mensagem_padrao,
        test_predict_sem_modelo_exibe_no_campo_detail,
        # Regra geral
        test_erro_422_nao_expoe_traceback,
        test_erro_400_nao_expoe_traceback,
        test_erro_500_nao_expoe_traceback,
    ]

    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  [OK] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {fn.__name__}: {exc}")
            failed += 1

    _cleanup()
    print(f"\nResultado: {passed}/{passed + failed} passou(aram)")
    if failed:
        sys.exit(1)
