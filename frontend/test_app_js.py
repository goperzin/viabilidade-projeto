"""Testes de estrutura do frontend/app.js (T-010)."""

import os
import re

JS_PATH = os.path.join(os.path.dirname(__file__), "app.js")


def read_js() -> str:
    with open(JS_PATH, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_function(js: str, name: str) -> bool:
    """Verifica se uma função nomeada está declarada (async ou não)."""
    patterns = [
        rf"async\s+function\s+{re.escape(name)}\s*\(",
        rf"function\s+{re.escape(name)}\s*\(",
        rf"const\s+{re.escape(name)}\s*=\s*(async\s*)?\(",
        rf"const\s+{re.escape(name)}\s*=\s*(async\s*)?function",
    ]
    return any(re.search(p, js) for p in patterns)


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_arquivo_existe():
    assert os.path.exists(JS_PATH), "app.js não encontrado"


def test_arquivo_nao_vazio():
    assert len(read_js().strip()) > 0, "app.js está vazio"


# --- Constante obrigatória (CLAUDE.md) ---

def test_api_base_declarado():
    js = read_js()
    assert "const API_BASE" in js, "Constante API_BASE ausente"


def test_api_base_aponta_para_localhost_8000():
    js = read_js()
    assert "http://localhost:8000" in js, "API_BASE deve apontar para http://localhost:8000"


def test_api_base_no_inicio_do_arquivo():
    js = read_js()
    lines = js.splitlines()
    # API_BASE deve aparecer nas primeiras 5 linhas não-vazias
    non_empty = [l.strip() for l in lines if l.strip()]
    first_five = "\n".join(non_empty[:5])
    assert "API_BASE" in first_five, "API_BASE deve estar no topo de app.js"


# --- Funções obrigatórias (SPRINT-001 + SPEC.md) ---

def test_funcao_checkStatus():
    assert has_function(read_js(), "checkStatus"), "Função checkStatus ausente"


def test_funcao_renderMetrics():
    assert has_function(read_js(), "renderMetrics"), "Função renderMetrics ausente"


def test_funcao_handleUpload():
    assert has_function(read_js(), "handleUpload"), "Função handleUpload ausente"


def test_funcao_handlePredict():
    assert has_function(read_js(), "handlePredict"), "Função handlePredict ausente"


def test_funcao_renderResult():
    assert has_function(read_js(), "renderResult"), "Função renderResult ausente"


def test_funcao_showError():
    assert has_function(read_js(), "showError"), "Função showError ausente"


def test_funcao_showLoading():
    assert has_function(read_js(), "showLoading"), "Função showLoading ausente"


def test_funcao_resetLoading():
    assert has_function(read_js(), "resetLoading"), "Função resetLoading ausente"


# --- Chamadas de API esperadas ---

def test_chama_get_status():
    js = read_js()
    assert "/status" in js, "Chamada para GET /status ausente"


def test_chama_post_train():
    js = read_js()
    assert "/train" in js, "Chamada para POST /train ausente"


def test_chama_post_predict():
    js = read_js()
    assert "/predict" in js, "Chamada para POST /predict ausente"


# --- Uso de fetch (sem frameworks) ---

def test_usa_fetch():
    js = read_js()
    assert "fetch(" in js or "fetch (" in js, "Deve usar fetch() nativo para chamadas HTTP"


def test_sem_frameworks_js():
    js = read_js().lower()
    for framework in ("import react", "require('react')", "require(\"react\")",
                      "import vue", "require('vue')", "require(\"vue\")"):
        assert framework not in js, f"Framework JS proibido detectado: {framework}"


# --- Loading state ---

def test_usa_showLoading_nos_handlers():
    js = read_js()
    # ambos os handlers devem chamar showLoading
    assert js.count("showLoading(") >= 2, \
        "showLoading deve ser chamado em pelo menos 2 handlers (upload e predict)"


def test_usa_resetLoading_nos_handlers():
    js = read_js()
    assert js.count("resetLoading(") >= 2, \
        "resetLoading deve ser chamado nos dois handlers"


# --- Formatação de métricas ---

def test_renderMetrics_formata_como_percentual():
    js = read_js()
    # Deve multiplicar por 100 para formatar 0.89 → "89%"
    assert "* 100" in js or "*100" in js, \
        "renderMetrics deve converter decimais para percentuais (* 100)"


# --- Event listeners ---

def test_registra_listener_upload_form():
    js = read_js()
    assert "upload-form" in js and "addEventListener" in js, \
        "Deve registrar addEventListener no upload-form"


def test_registra_listener_predict_form():
    js = read_js()
    assert "predict-form" in js and "addEventListener" in js, \
        "Deve registrar addEventListener no predict-form"


def test_chama_checkStatus_ao_inicializar():
    js = read_js()
    # checkStatus() sem await no nível de módulo (chamada de inicialização)
    assert re.search(r"checkStatus\(\)\s*;?\s*$", js, re.MULTILINE), \
        "checkStatus() deve ser chamado ao final do arquivo para inicializar"
