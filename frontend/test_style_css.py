"""Testes de estrutura do frontend/style.css (T-009)."""

import os
import re

CSS_PATH = os.path.join(os.path.dirname(__file__), "style.css")


def read_css() -> str:
    with open(CSS_PATH, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def has_selector(css: str, selector: str) -> bool:
    """Verifica se um seletor existe no CSS (sem considerar aninhamento)."""
    pattern = re.escape(selector) + r"\s*\{"
    return bool(re.search(pattern, css))


def has_var(css: str, var_name: str) -> bool:
    """Verifica se uma variável CSS está declarada no :root."""
    return var_name + ":" in css


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_arquivo_existe():
    assert os.path.exists(CSS_PATH), "style.css não encontrado"


def test_arquivo_nao_vazio():
    assert len(read_css().strip()) > 0, "style.css está vazio"


# --- Variáveis no :root ---

def test_root_declarado():
    css = read_css()
    assert ":root" in css, "Bloco :root ausente"


def test_variaveis_cores_principais():
    css = read_css()
    variaveis = [
        "--color-bg",
        "--color-surface",
        "--color-primary",
        "--color-text",
        "--color-border",
        "--color-error",
        "--color-viable",
        "--color-inviable",
    ]
    for v in variaveis:
        assert has_var(css, v), f"Variável {v} ausente no CSS"


def test_variaveis_espacamento():
    css = read_css()
    for v in ["--space-sm", "--space-md", "--space-lg", "--space-xl"]:
        assert has_var(css, v), f"Variável {v} ausente no CSS"


# --- Classes semânticas obrigatórias (CLAUDE.md) ---

def test_classe_card():
    css = read_css()
    assert has_selector(css, ".card"), "Seletor .card ausente"


def test_classe_badge_viable():
    css = read_css()
    assert has_selector(css, ".badge--viable"), "Seletor .badge--viable ausente"


def test_classe_badge_inviable():
    css = read_css()
    assert has_selector(css, ".badge--inviable"), "Seletor .badge--inviable ausente"


def test_classe_metrics_panel():
    css = read_css()
    assert ".metrics-panel" in css, "Seletor .metrics-panel ausente"


def test_classe_metrics_grid():
    css = read_css()
    assert ".metrics-grid" in css, "Seletor .metrics-grid ausente"


def test_classe_progress_bar():
    css = read_css()
    assert has_selector(css, ".progress-bar"), "Seletor .progress-bar ausente"


def test_classe_progress_bar_fill():
    css = read_css()
    assert ".progress-bar__fill" in css, "Seletor .progress-bar__fill ausente"


def test_classe_btn_loading():
    css = read_css()
    assert ".btn--loading" in css, "Seletor .btn--loading ausente"


# --- Layout ---

def test_max_width_definido():
    css = read_css()
    assert "max-width" in css, "Propriedade max-width ausente"


def test_classe_container():
    css = read_css()
    assert has_selector(css, ".container"), "Seletor .container ausente"


# --- Mobile-first: breakpoint min-width: 768px ---

def test_breakpoint_tablet_min_width_768():
    css = read_css()
    assert "min-width: 768px" in css or "min-width:768px" in css, \
        "Breakpoint min-width: 768px ausente"


def test_media_query_usa_min_width():
    css = read_css()
    # mobile-first exige min-width, não max-width
    media_blocks = re.findall(r"@media[^{]+\{", css)
    for block in media_blocks:
        assert "min-width" in block, \
            f"Media query não é mobile-first (usa max-width): {block.strip()}"


# --- Formulários ---

def test_classe_form_group():
    css = read_css()
    assert ".form-group" in css, "Seletor .form-group ausente"


def test_classe_form_input():
    css = read_css()
    assert has_selector(css, ".form-input"), "Seletor .form-input ausente"


def test_classe_btn_primary():
    css = read_css()
    assert has_selector(css, ".btn--primary"), "Seletor .btn--primary ausente"


# --- Error banner ---

def test_classe_error_banner():
    css = read_css()
    assert has_selector(css, ".error-banner"), "Seletor .error-banner ausente"
