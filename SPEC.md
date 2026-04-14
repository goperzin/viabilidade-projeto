# SPEC.md — Especificação Técnica

## 1. Stack Tecnológica

| Camada | Tecnologia |
|--------|-----------|
| Frontend | HTML5 + CSS3 + JavaScript (ES6+) puro |
| Backend | Python 3.11+ com FastAPI |
| ML | scikit-learn, pandas, numpy, joblib |
| Servidor | Uvicorn |
| Comunicação | REST API + JSON |

---

## 2. Estrutura de Pastas

```
project-viability/
│
├── backend/
│   ├── main.py                  # App FastAPI + rotas
│   ├── model.py                 # Lógica de treino e previsão
│   ├── schemas.py               # Modelos Pydantic (request/response)
│   ├── requirements.txt
│   └── artifacts/               # Modelos salvos (gitignored)
│       ├── logistic_model.joblib
│       ├── logistic_model_scaler.joblib
│       └── logistic_model_metrics.joblib
│
├── frontend/
│   ├── index.html               # Página principal
│   ├── style.css                # Estilos globais
│   └── app.js                   # Lógica JS + chamadas fetch
│
├── PRD.md
├── SPEC.md
├── CLAUDE.md
└── SPRINT-001.md
```

---

## 3. API — Endpoints

### `GET /status`
Verifica se o modelo está treinado.

**Response 200:**
```json
{
  "model_trained": true,
  "metrics": {
    "0": { "precision": 0.91, "recall": 0.88, "f1-score": 0.89 },
    "1": { "precision": 0.87, "recall": 0.90, "f1-score": 0.88 },
    "accuracy": 0.89
  }
}
```

**Response quando não treinado:**
```json
{
  "model_trained": false,
  "metrics": null
}
```

---

### `POST /train`
Recebe um CSV e treina o modelo.

**Request:** `multipart/form-data`  
- `file`: arquivo `.csv`

**Validação do CSV:**  
Colunas obrigatórias: `investment`, `expected_return`, `impact_score`, `viability`

**Response 200:**
```json
{
  "success": true,
  "message": "Modelo treinado com sucesso.",
  "metrics": { ... }
}
```

**Response 422 (colunas faltando):**
```json
{
  "success": false,
  "message": "Colunas ausentes: ['viability']"
}
```

---

### `POST /predict`
Recebe dados de um projeto e retorna previsão.

**Request body:**
```json
{
  "investment": 40000,
  "expected_return": 60000,
  "impact_score": 6
}
```

**Response 200:**
```json
{
  "viability": 1,
  "viability_label": "Viável",
  "probability": 0.83
}
```

**Response 400 (modelo não treinado):**
```json
{
  "detail": "Modelo ainda não foi treinado. Faça upload de um CSV primeiro."
}
```

---

## 4. Schemas Pydantic (`schemas.py`)

```python
class ProjectInput(BaseModel):
    investment: float
    expected_return: float
    impact_score: float

class PredictionResponse(BaseModel):
    viability: int
    viability_label: str
    probability: float

class TrainResponse(BaseModel):
    success: bool
    message: str
    metrics: dict | None
```

---

## 5. Lógica de ML (`model.py`)

- `train_model(df: pd.DataFrame) -> dict` — treina e salva modelo + scaler + métricas
- `predict(project: dict) -> dict` — carrega modelo, normaliza input, retorna previsão
- `get_metrics() -> dict | None` — retorna métricas salvas ou None
- `model_exists() -> bool` — verifica se artefatos existem em disco

---

## 6. Frontend — Componentes JS

```
app.js
├── checkStatus()         → GET /status ao carregar a página
├── renderMetrics(data)   → Renderiza painel de métricas
├── handleUpload()        → POST /train com FormData
├── handlePredict()       → POST /predict com dados do formulário
└── renderResult(data)    → Exibe card de resultado com probabilidade
```

---

## 7. Tratamento de Erros

| Cenário | Comportamento |
|---------|--------------|
| CSV sem colunas obrigatórias | Mensagem de erro inline no upload |
| Modelo não treinado ao prever | Banner de aviso + redirect para upload |
| Campos inválidos no formulário | Validação client-side antes do fetch |
| Erro 500 da API | Toast de erro genérico na tela |

---

## 8. CORS

Configurado no FastAPI para aceitar `http://localhost:*` em desenvolvimento.

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir em produção
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 9. Como Rodar Localmente

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
# Abrir frontend/index.html diretamente no browser
# ou usar live-server / qualquer servidor estático
```
