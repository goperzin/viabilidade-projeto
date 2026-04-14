"""Testes de estrutura do frontend/index.html (T-008)."""

import os
from html.parser import HTMLParser


HTML_PATH = os.path.join(os.path.dirname(__file__), "index.html")


# ---------------------------------------------------------------------------
# Parser auxiliar — coleta ids, classes, tags e atributos
# ---------------------------------------------------------------------------

class HTMLInspector(HTMLParser):
    """Coleta todos os elementos do HTML para inspeção."""

    def __init__(self):
        super().__init__()
        self.ids: set[str] = set()
        self.classes: set[str] = set()
        self.tags: list[str] = []
        self.attrs_by_id: dict[str, dict] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple]) -> None:
        self.tags.append(tag)
        attr_dict = dict(attrs)

        if "id" in attr_dict:
            elem_id = attr_dict["id"]
            self.ids.add(elem_id)
            self.attrs_by_id[elem_id] = attr_dict

        if "class" in attr_dict:
            for cls in attr_dict["class"].split():
                self.classes.add(cls)


def parse_html() -> HTMLInspector:
    inspector = HTMLInspector()
    with open(HTML_PATH, encoding="utf-8") as f:
        inspector.feed(f.read())
    return inspector


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_arquivo_existe():
    assert os.path.exists(HTML_PATH), "index.html não encontrado"


def test_charset_utf8():
    with open(HTML_PATH, encoding="utf-8") as f:
        content = f.read()
    assert 'charset="UTF-8"' in content or "charset='UTF-8'" in content.lower()


def test_referencia_style_css():
    with open(HTML_PATH, encoding="utf-8") as f:
        content = f.read()
    assert "style.css" in content


def test_referencia_app_js():
    with open(HTML_PATH, encoding="utf-8") as f:
        content = f.read()
    assert "app.js" in content


def test_secao_upload_existe():
    inspector = parse_html()
    assert "upload-section" in inspector.ids, "Seção #upload-section ausente"


def test_secao_upload_tem_hidden():
    inspector = parse_html()
    attrs = inspector.attrs_by_id.get("upload-section", {})
    assert "hidden" in attrs, "#upload-section deve ter atributo hidden por padrão"


def test_secao_metricas_existe():
    inspector = parse_html()
    assert "metrics-section" in inspector.ids, "Seção #metrics-section ausente"


def test_secao_metricas_tem_hidden():
    inspector = parse_html()
    attrs = inspector.attrs_by_id.get("metrics-section", {})
    assert "hidden" in attrs, "#metrics-section deve ter atributo hidden por padrão"


def test_secao_formulario_predict_existe():
    inspector = parse_html()
    assert "predict-section" in inspector.ids, "Seção #predict-section ausente"


def test_secao_formulario_tem_hidden():
    inspector = parse_html()
    attrs = inspector.attrs_by_id.get("predict-section", {})
    assert "hidden" in attrs, "#predict-section deve ter atributo hidden por padrão"


def test_secao_resultado_existe():
    inspector = parse_html()
    assert "result-section" in inspector.ids, "Seção #result-section ausente"


def test_secao_resultado_tem_hidden():
    inspector = parse_html()
    attrs = inspector.attrs_by_id.get("result-section", {})
    assert "hidden" in attrs, "#result-section deve ter atributo hidden por padrão"


def test_badge_status_existe():
    inspector = parse_html()
    assert "status-badge" in inspector.ids, "Elemento #status-badge ausente no header"


def test_formulario_predict_tem_campos_obrigatorios():
    inspector = parse_html()
    assert "investment" in inspector.ids, "Campo #investment ausente"
    assert "expected_return" in inspector.ids, "Campo #expected_return ausente"
    assert "impact_score" in inspector.ids, "Campo #impact_score ausente"


def test_formulario_upload_existe():
    inspector = parse_html()
    assert "upload-form" in inspector.ids, "Formulário #upload-form ausente"
    assert "csv-file" in inspector.ids, "Input de arquivo #csv-file ausente"


def test_botao_upload_existe():
    inspector = parse_html()
    assert "upload-btn" in inspector.ids, "Botão #upload-btn ausente"


def test_botao_predict_existe():
    inspector = parse_html()
    assert "predict-btn" in inspector.ids, "Botão #predict-btn ausente"


def test_progress_bar_existe():
    inspector = parse_html()
    assert "progress-bar" in inspector.classes, "Elemento .progress-bar ausente"
    assert "progress-fill" in inspector.ids, "Elemento #progress-fill ausente"


def test_metrics_grid_existe():
    inspector = parse_html()
    assert "metrics-grid" in inspector.ids, "Elemento #metrics-grid ausente"


def test_error_banner_existe():
    inspector = parse_html()
    assert "error-banner" in inspector.ids, "Elemento #error-banner ausente"
    assert "error-message" in inspector.ids, "Elemento #error-message ausente"


def test_lang_pt_br():
    with open(HTML_PATH, encoding="utf-8") as f:
        content = f.read()
    assert 'lang="pt-BR"' in content, 'Atributo lang="pt-BR" ausente na tag <html>'
