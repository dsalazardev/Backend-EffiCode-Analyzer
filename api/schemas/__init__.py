# Schemas Package
# Contiene todos los modelos Pydantic para request/response

from .analysis import AnalysisRequest, AnalysisResponse

__all__ = [
    "AnalysisRequest",
    "AnalysisResponse",
]
