# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Contexto do Projeto

Aplicação web de classificação de viabilidade de projetos usando **Regressão Logística** (scikit-learn).
Features: `investment`, `expected_return`, `impact_score` → target: `viability` (0 ou 1).

Leia **PRD.md** para requisitos e **SPEC.md** para contratos de API e arquitetura antes de gerar código.

---

## Comandos

```bash
# Instalar dependências do backend
cd backend && pip install -r requirements.txt

# Rodar o backend (porta 8000, hot reload)
cd backend && uvicorn main:app --reload --port 8000

# Testar endpoints manualmente
curl http://localhost:8000/status
curl -X POST http://localhost:8000/train -F "file=@projects_data.csv"
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"investment": 40000, "expected_return": 60000, "impact_score": 6}'

# Frontend: abrir diretamente no browser
# frontend/index.html (ou usar live-server / qualquer servidor estático)
```

---

## Arquitetura

```
backend/
  main.py       # FastAPI app: rotas + CORS + validação de entrada
  model.py      # TODA a lógica ML (treino, previsão, métricas)
  schemas.py    # Pydantic schemas: ProjectInput, PredictionResponse, TrainResponse, StatusResponse
  artifacts/    # Artefatos joblib (gitignored): modelo, scaler, métricas
frontend/
  index.html    # SPA de página única com seções condicionais (upload vs formulário)
  style.css     # CSS puro com variáveis no :root
  app.js        # fetch async/await, sem frameworks
```

**Fluxo de dados:** Frontend checa `GET /status` ao carregar → exibe upload (modelo ausente) ou formulário (modelo presente) → `POST /train` treina e persiste 3 arquivos `.joblib` → `POST /predict` carrega artefatos, normaliza input com o scaler salvo, retorna viabilidade + probabilidade.

**Persistência:** exclusivamente via arquivos `.joblib` em `backend/artifacts/`. Sem banco de dados.

---

## Stack

- **Backend:** Python 3.11+ · FastAPI · Uvicorn
- **ML:** scikit-learn · pandas · numpy · joblib
- **Frontend:** HTML5 · CSS3 · JavaScript ES6+ puro (sem frameworks)

---

## Regras

### Sempre
- Lógica de ML exclusivamente em `model.py` — `main.py` só orquestra rotas
- Type hints em todo o Python; docstrings curtas em português
- Schemas Pydantic para todos os request/response bodies
- Validar CSV (extensão + colunas obrigatórias) antes de treinar
- Paths via `os.path.join` usando as constantes em `model.py`, nunca hardcodados
- Erros em português; nunca expor stack traces na response da API

### Nunca
- Frameworks JS (React, Vue, etc.)
- Banco de dados (SQLite, PostgreSQL, etc.)

---

## Convenções de Código

### Python — constantes de path em `model.py`
```python
ARTIFACTS_DIR = "artifacts"
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "logistic_model.joblib")
SCALER_PATH   = os.path.join(ARTIFACTS_DIR, "logistic_model_scaler.joblib")
METRICS_PATH  = os.path.join(ARTIFACTS_DIR, "logistic_model_metrics.joblib")

REQUIRED_COLUMNS = {"investment", "expected_return", "impact_score", "viability"}
```

### JavaScript
- `const API_BASE = 'http://localhost:8000'` no topo de `app.js`
- Separar funções de renderização das de chamada de API
- Loading state em todo botão que dispara fetch

### CSS
- Variáveis no `:root` para cores e espaçamentos
- Mobile-first com breakpoints `min-width`
- Classes semânticas: `.card`, `.badge--viable`, `.badge--inviable`, `.metrics-panel`, `.progress-bar`

---

## Mensagens de Erro Padrão

| Situação | Mensagem |
|----------|----------|
| Modelo não treinado | `"Modelo ainda não foi treinado. Faça upload de um CSV primeiro."` |
| Colunas ausentes no CSV | `"Colunas ausentes no CSV: {lista}"` |
| Arquivo inválido | `"Apenas arquivos .csv são aceitos."` |
| Erro interno | `"Erro interno ao processar a requisição."` |

---

## Ordem de Implementação

```
T-001  Estrutura de pastas
T-002  backend/model.py
T-003  backend/schemas.py
T-004  backend/main.py  (GET /status)
T-005  backend/main.py  (POST /train)
T-006  backend/main.py  (POST /predict)
T-007  CORS + teste de todos os endpoints
T-008  frontend/index.html
T-009  frontend/style.css
T-010  frontend/app.js
T-011  Teste do fluxo completo
T-012  Teste dos casos de erro
```

Siga `SPRINT-001.md` e marque cada task como `[x]` ao concluir.

---

## Estado Atual

**Sprint:** SPRINT-001 — em andamento  
**Última atualização:** 2026-04-14

### Tarefas concluídas

| Task | Descrição |
|------|-----------|
| T-001 | Estrutura de pastas criada conforme SPEC.md |
| T-002 | `model.py` implementado — `model_exists`, `train_model`, `predict`, `get_metrics` (9 testes passando) |
| T-003 | `schemas.py` implementado — `ProjectInput`, `PredictionResponse`, `TrainResponse`, `StatusResponse` (13 testes passando) |
| T-004 | `GET /status` implementado em `main.py` com CORS e estrutura base do app (6 testes passando) |
| T-005 | `POST /train` testado e corrigido em `main.py` — validação de extensão, colunas e treino (10 testes passando) |
| T-006 | `POST /predict` testado em `main.py` — 400 sem modelo, fluxo feliz, validação de entrada (13 testes passando) |
| T-007 | CORS verificado e todos os endpoints testados — cabeçalhos, preflight OPTIONS, fluxo E2E (8 testes passando) |
| T-008 | `index.html` estruturado — header com badge, seção upload, painel métricas, formulário predict, área resultado (21 testes passando) |

### Pendentes (próxima execução)

| Task | Descrição |
|------|-----------|
| T-009 | Implementar `style.css` |
| T-010 | Implementar `app.js` |
| T-011 | Teste do fluxo completo (end-to-end) |
| T-012 | Teste dos casos de erro |

### Decisões técnicas tomadas (não previstas no SPEC original)

- **`.gitkeep` em `backend/artifacts/`** — necessário para versionar a pasta vazia no git, já que o `.gitignore` exclui os `.joblib`. O SPEC não mencionava como preservar o diretório.
- **`.gitignore` criado na raiz** — o SPEC indica que `artifacts/` é gitignored, mas não especificava onde ou como configurar o arquivo. Optou-se por criar na raiz com escopo `backend/artifacts/*.joblib`, `__pycache__/` e `*.pyc`.
- **`Field(...)` com descriptions em `schemas.py`** — o SPEC define apenas os tipos; optou-se por adicionar `Field` com `description` em português para melhorar a documentação automática do Swagger gerado pelo FastAPI.
- **`metrics: dict | None` com `default=None`** — `TrainResponse` e `StatusResponse` usam `Field(None, ...)` para tornar `metrics` opcional por padrão, evitando que o caller precise passar `metrics=None` explicitamente.
- **`main.py` criado completo na T-004** — o SPEC divide as rotas em T-004/T-005/T-006, mas como `main.py` é um único arquivo de app, foi criado já com os três endpoints para evitar estados intermediários inválidos. Os testes de T-004 cobrem apenas `GET /status`; os de T-005 e T-006 cobrirão seus respectivos endpoints.
- **`httpx` adicionado como dependência de teste** — `FastAPI.TestClient` exige `httpx` (não declarado no `requirements.txt` original). Instalado separadamente; pode ser adicionado ao `requirements.txt` como `httpx` em dev-dependencies se o projeto crescer.
- **Erros de leitura de CSV usam HTTP 500, não 422** — falha ao parsear o arquivo (CSV malformado, binário etc.) é erro interno, não de validação do cliente. Erros de schema (extensão, colunas ausentes) continuam como 422.
- **`import io` movido para o topo do módulo** — estava dentro do handler `async def train`; corrigido para seguir convenção PEP 8 de imports no topo do arquivo.
- **FastAPI serializa JSON como UTF-8 raw bytes (`ensure_ascii=False`)** — a label "Inviável" chega ao cliente como bytes UTF-8 corretos (0xC3 0xA1). A exibição errada via `curl | json.tool` no Windows (cp1252) era artefato do terminal, não bug da API. Confirmado via TestClient com verificação de codepoints.
- **`python -m uvicorn` em vez de `uvicorn` diretamente** — no Windows com Python em `AppData/Roaming`, o executável `uvicorn` não é adicionado ao PATH automaticamente. Usar `python -m uvicorn main:app --reload --port 8000` no diretório `backend/`.
- **Testes de `index.html` em `frontend/test_index_html.py`** — como não há framework de testes JS no projeto, os testes do HTML foram escritos em Python usando `html.parser` nativo (sem dependências externas). Verificam presença de IDs, classes, atributo `hidden` e referências a `style.css` e `app.js`.
