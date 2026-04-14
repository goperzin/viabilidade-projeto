"""FastAPI app — rotas, CORS e validacao de entrada para a API de viabilidade."""

import io
import os

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import model as ml
from schemas import ProjectInput, PredictionResponse, StatusResponse, TrainResponse

# ---------------------------------------------------------------------------
# Configuracao do app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de Viabilidade de Projetos",
    description="Classifica projetos como viaveis ou inviaveis usando Regressao Logistica.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restringir em producao
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse, summary="Status do modelo")
def get_status() -> StatusResponse:
    """Retorna se o modelo esta treinado e as metricas salvas."""
    return StatusResponse(
        model_trained=ml.model_exists(),
        metrics=ml.get_metrics(),
    )


@app.post("/train", response_model=TrainResponse, summary="Treinar modelo com CSV")
async def train(file: UploadFile = File(...)) -> TrainResponse:
    """Recebe um CSV, valida e treina o modelo de regressao logistica."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Apenas arquivos .csv sao aceitos.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno ao processar a requisicao.")

    missing = ml.REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Colunas ausentes no CSV: {sorted(missing)}",
        )

    try:
        metrics = ml.train_model(df)
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno ao processar a requisicao.")

    return TrainResponse(
        success=True,
        message="Modelo treinado com sucesso.",
        metrics=metrics,
    )


@app.post("/predict", response_model=PredictionResponse, summary="Prever viabilidade")
def predict(project: ProjectInput) -> PredictionResponse:
    """Recebe dados de um projeto e retorna a previsao de viabilidade."""
    if not ml.model_exists():
        raise HTTPException(
            status_code=400,
            detail="Modelo ainda nao foi treinado. Faca upload de um CSV primeiro.",
        )

    try:
        result = ml.predict(project.model_dump())
    except Exception:
        raise HTTPException(status_code=500, detail="Erro interno ao processar a requisicao.")

    return PredictionResponse(**result)
