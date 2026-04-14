"""Schemas Pydantic para request/response bodies da API de viabilidade."""

from pydantic import BaseModel, Field


class ProjectInput(BaseModel):
    """Dados de entrada para previsão de viabilidade de um projeto."""

    investment:      float = Field(..., description="Valor do investimento em reais")
    expected_return: float = Field(..., description="Retorno esperado em reais")
    impact_score:    float = Field(..., description="Pontuacao de impacto (1-10)")


class PredictionResponse(BaseModel):
    """Resposta da previsao de viabilidade."""

    viability:       int   = Field(..., description="Classificacao: 1 (viavel) ou 0 (inviavel)")
    viability_label: str   = Field(..., description="Label legivel: 'Viavel' ou 'Inviavel'")
    probability:     float = Field(..., description="Probabilidade da classificacao (0.0-1.0)")


class TrainResponse(BaseModel):
    """Resposta do endpoint de treino do modelo."""

    success: bool       = Field(..., description="Indica se o treino foi bem-sucedido")
    message: str        = Field(..., description="Mensagem descritiva do resultado")
    metrics: dict | None = Field(None, description="Metricas do modelo treinado ou null")


class StatusResponse(BaseModel):
    """Resposta do endpoint de status do modelo."""

    model_trained: bool       = Field(..., description="True se o modelo esta treinado")
    metrics:       dict | None = Field(None, description="Metricas do modelo ou null")
