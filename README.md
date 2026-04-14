# Project Viability Classifier

Aplicação web para classificação de viabilidade de projetos usando **Regressão Logística**. O usuário faz upload de um CSV com dados históricos, treina o modelo e obtém previsões com probabilidade e métricas de desempenho — tudo via interface web, sem código.

---

## Demonstração

| Estado inicial (sem modelo) | Após treino |
|---|---|
| Exibe área de upload de CSV | Exibe métricas + formulário de previsão |

**Fluxo:**
```
Abrir app → Upload CSV → Treinar modelo → Preencher formulário → Ver resultado (Viável / Inviável + %)
```

---

## Funcionalidades

- Upload de CSV para treino do modelo de Regressão Logística
- Painel de métricas: Accuracy, Precision, Recall e F1-Score
- Formulário de previsão com validação client-side
- Resultado visual com classificação e barra de probabilidade
- Persistência do modelo em disco entre sessões (sem banco de dados)
- Interface responsiva (mobile-first)

---

## Stack

| Camada | Tecnologia |
|--------|-----------|
| Frontend | HTML5 + CSS3 + JavaScript ES6+ puro |
| Backend | Python 3.11+ · FastAPI · Uvicorn |
| ML | scikit-learn · pandas · numpy · joblib |
| Comunicação | REST API + JSON |

---

## Estrutura do Projeto

```
viabilidade-projeto/
├── backend/
│   ├── main.py              # Rotas FastAPI + CORS
│   ├── model.py             # Lógica de treino e previsão (scikit-learn)
│   ├── schemas.py           # Schemas Pydantic (request/response)
│   ├── requirements.txt
│   └── artifacts/           # Artefatos do modelo (gitignored)
├── frontend/
│   ├── index.html           # SPA com seções condicionais
│   ├── style.css            # Design system com variáveis CSS
│   └── app.js               # Fetch async/await, sem frameworks
├── projects_data.csv        # Dataset de exemplo (100 linhas)
├── PRD.md                   # Requisitos do produto
└── SPEC.md                  # Especificação técnica da API
```

---

## Pré-requisitos

- Python 3.11+
- pip

---

## Instalação e Execução

### 1. Clonar o repositório

```bash
git clone https://github.com/goperzin/viabilidade-projeto.git
cd viabilidade-projeto
```

### 2. Instalar dependências do backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Iniciar o backend

```bash
# Linux / macOS
uvicorn main:app --reload --port 8000

# Windows (caso uvicorn não esteja no PATH)
python -m uvicorn main:app --reload --port 8000
```

A API estará disponível em `http://localhost:8000`.

### 4. Abrir o frontend

Abra `frontend/index.html` diretamente no browser ou use um servidor estático:

```bash
# Com live-server (npm)
npx live-server frontend

# Com Python
python -m http.server 3000 --directory frontend
```

---

## API

### `GET /status`
Verifica se o modelo está treinado.

```json
{
  "model_trained": true,
  "metrics": {
    "accuracy": 0.89,
    "0": { "precision": 0.91, "recall": 0.88, "f1-score": 0.89 },
    "1": { "precision": 0.87, "recall": 0.90, "f1-score": 0.88 }
  }
}
```

### `POST /train`
Treina o modelo com um arquivo CSV.

```bash
curl -X POST http://localhost:8000/train \
  -F "file=@projects_data.csv"
```

Colunas obrigatórias no CSV: `investment`, `expected_return`, `impact_score`, `viability`

### `POST /predict`
Retorna a previsão de viabilidade para um projeto.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"investment": 40000, "expected_return": 60000, "impact_score": 6}'
```

```json
{
  "viability": 1,
  "viability_label": "Viável",
  "probability": 0.83
}
```

---

## Dataset de Exemplo

O arquivo `projects_data.csv` na raiz contém 100 projetos de exemplo (gerado com seed 42).

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `investment` | float | Valor investido no projeto |
| `expected_return` | float | Retorno esperado |
| `impact_score` | float | Pontuação de impacto (0–10) |
| `viability` | int | 0 = Inviável · 1 = Viável |

---

## Testes

Os testes cobrem model, schemas, endpoints, frontend e fluxo E2E.

```bash
cd backend
python -m pytest -v
```

```bash
cd frontend
python -m pytest -v
```

---

## Licença

Este projeto está sob a licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.
