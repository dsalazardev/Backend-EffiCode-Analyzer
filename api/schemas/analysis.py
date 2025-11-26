"""
Schemas para el módulo de análisis de complejidad.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class AnalysisRequest(BaseModel):
    """Request para analizar pseudocódigo."""
    pseudocode: str = Field(
        ..., 
        description="Pseudocódigo estilo Cormen a analizar",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "pseudocode": """INSERTION-SORT(A, n)
    for j ← 2 to n do
        key ← A[j]
        i ← j - 1
        while i > 0 and A[i] > key do
            A[i + 1] ← A[i]
            i ← i - 1
        A[i + 1] ← key
    return A"""
            }
        }


class AnalysisResponse(BaseModel):
    """Response con el resultado del análisis de complejidad."""
    complexity_o: str = Field(..., description="Notación Big O (peor caso)")
    complexity_omega: str = Field(..., description="Notación Big Omega (mejor caso)")
    complexity_theta: str = Field(..., description="Notación Big Theta (caso promedio)")
    justification: str = Field(..., description="Justificación matemática")
    justification_data: Dict[str, Any] = Field(..., description="Datos estructurados de la justificación")
    validation: str = Field(..., description="Validación por IA")
    ast_image: str = Field(..., description="Imagen del AST en Base64 (PNG)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "complexity_o": "O(n²)",
                "complexity_omega": "Ω(n)",
                "complexity_theta": "No aplicable",
                "justification": "El algoritmo contiene bucles anidados...",
                "justification_data": {},
                "validation": "El análisis es correcto...",
                "ast_image": "data:image/png;base64,..."
            }
        }
