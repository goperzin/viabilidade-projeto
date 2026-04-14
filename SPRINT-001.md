# SPRINT-001.md — Fundação do Projeto

**Período:** Sprint 1  
**Objetivo:** Ter a aplicação funcionando end-to-end: upload de CSV → treino do modelo → previsão via formulário → exibição de resultado

---

## Critério de Conclusão da Sprint

A sprint está completa quando:
- O usuário consegue fazer upload de um CSV e ver as métricas do modelo treinado
- O usuário consegue preencher o formulário e ver o resultado de viabilidade com probabilidade
- Não há erros não tratados visíveis na interface

---

## Tarefas

### 🗂️ Estrutura

- [x] **T-001** — Criar estrutura de pastas conforme SPEC.md
  - `backend/`, `backend/artifacts/`, `frontend/`
  - Arquivos vazios: `main.py`, `model.py`, `schemas.py`, `requirements.txt`, `index.html`, `style.css`, `app.js`

---

### ⚙️ Backend

- [x] **T-002** — Implementar `model.py`
  - Constantes de path (`MODEL_PATH`, `SCALER_PATH`, `METRICS_PATH`)
  - `model_exists() -> bool`
  - `train_model(df: pd.DataFrame) -> dict` — treina, salva e retorna métricas
  - `predict(project: dict) -> dict` — carrega modelo, normaliza, retorna viabilidade + probabilidade
  - `get_metrics() -> dict | None`

- [x] **T-003** — Implementar `schemas.py`
  - `ProjectInput` (investment, expected_return, impact_score)
  - `PredictionResponse` (viability, viability_label, probability)
  - `TrainResponse` (success, message, metrics)
  - `StatusResponse` (model_trained, metrics)

- [x] **T-004** — Implementar `GET /status` em `main.py`
  - Retorna `model_trained: true/false` + métricas (ou null)

- [x] **T-005** — Implementar `POST /train` em `main.py`
  - Recebe `multipart/form-data` com campo `file`
  - Valida extensão `.csv`
  - Valida colunas obrigatórias
  - Chama `train_model()` e retorna `TrainResponse`

- [ ] **T-006** — Implementar `POST /predict` em `main.py`
  - Valida que modelo existe (400 se não)
  - Chama `predict()` e retorna `PredictionResponse`

- [ ] **T-007** — Configurar CORS e rodar com Uvicorn
  - Testar todos os endpoints com curl ou Postman

---

### 🎨 Frontend

- [ ] **T-008** — Estruturar `index.html`
  - Seção de status do modelo (header com badge)
  - Seção de upload de CSV (visível apenas se modelo não treinado)
  - Painel de métricas (visível após treino)
  - Formulário de previsão (investment, expected_return, impact_score)
  - Área de resultado da previsão

- [ ] **T-009** — Implementar `style.css`
  - Variáveis CSS (`:root`) para cores e espaçamentos
  - Layout centralizado com max-width
  - Card component (`.card`)
  - Badges de viabilidade (`.badge--viable` verde, `.badge--inviable` vermelho)
  - Barra de probabilidade (`.progress-bar`)
  - Painel de métricas (`.metrics-panel`, `.metrics-grid`)
  - Estados de loading (`.btn--loading`)
  - Responsivo para tablet (min-width: 768px)

- [ ] **T-010** — Implementar `app.js`
  - Constante `API_BASE = 'http://localhost:8000'`
  - `checkStatus()` — chama `GET /status`, mostra/oculta seções conforme resultado
  - `renderMetrics(metrics)` — preenche painel de métricas dinamicamente
  - `handleUpload(e)` — envia CSV via FormData para `POST /train`, exibe feedback
  - `handlePredict(e)` — lê formulário, envia para `POST /predict`, chama `renderResult`
  - `renderResult(data)` — exibe card com viabilidade, label e barra de probabilidade
  - `showError(msg)` / `showLoading(btn)` / `resetLoading(btn)` — utilitários de UI

---

### ✅ Testes Manuais

- [ ] **T-011** — Teste completo do fluxo feliz
  1. Abrir `index.html` no browser
  2. Fazer upload de `projects_data.csv` válido
  3. Confirmar que métricas aparecem
  4. Preencher formulário com `investment: 40000`, `expected_return: 60000`, `impact_score: 6`
  5. Confirmar resultado com probabilidade

- [ ] **T-012** — Teste de casos de erro
  - Upload de arquivo não-CSV → mensagem de erro
  - Upload de CSV sem coluna `viability` → mensagem de erro
  - Campos do formulário em branco → validação client-side
  - Acessar `/predict` sem modelo treinado → mensagem de aviso

---

## Dependências Python (`requirements.txt`)

```
fastapi
uvicorn[standard]
python-multipart
pandas
scikit-learn
numpy
joblib
```

---

## Notas para o Claude Code

- Implemente na ordem das tasks (T-001 → T-012)
- Não pule para o frontend antes do backend estar testado
- O arquivo `projects_data.csv` de exemplo deve ter ao menos 50 linhas para o treino ser representativo
- Ao exibir métricas, formatar valores como `0.89` → `89%`
