from __future__ import annotations
from typing import TYPE_CHECKING, Optional

# Importaciones de tipos para evitar dependencias circulares
if TYPE_CHECKING:
    from .Algoritmo import Algoritmo
    from .Complejidad import Complejidad
    from .Analizador import Analizador


class Reporte:
    """
    Consolida el resultado de un análisis de complejidad.
    Mapea la clase 'Reporte' del diagrama UML.
    """

    def __init__(self, id: int, algoritmo_analizado: Algoritmo, resultado_complejidad: Complejidad):
        self._id = id
        self._algoritmo_analizado = algoritmo_analizado
        self._resultado_complejidad = resultado_complejidad
        self._validacion_llm: str = ""
        self._algoritmo: Optional[Algoritmo] = None  # Relación bidireccional
        self._analizador: Optional[Analizador] = None

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    def algoritmo_analizado(self) -> Algoritmo:
        return self._algoritmo_analizado

    @algoritmo_analizado.setter
    def algoritmo_analizado(self, value: Algoritmo):
        self._algoritmo_analizado = value

    @property
    def resultado_complejidad(self) -> Complejidad:
        return self._resultado_complejidad

    @resultado_complejidad.setter
    def resultado_complejidad(self, value: Complejidad):
        self._resultado_complejidad = value

    @property
    def validacion_llm(self) -> str:
        return self._validacion_llm

    @validacion_llm.setter
    def validacion_llm(self, value: str):
        self._validacion_llm = value

    @property
    def algoritmo(self) -> Optional[Algoritmo]:
        return self._algoritmo

    @algoritmo.setter
    def algoritmo(self, value: Optional[Algoritmo]):
        self._algoritmo = value

    @property
    def analizador(self) -> Optional[Analizador]:
        return self._analizador

    @analizador.setter
    def analizador(self, value: Optional[Analizador]):
        self._analizador = value

    def addAlgoritmo(self, algoritmo: Algoritmo):
        self._algoritmo = algoritmo

    def removeAlgoritmo(self):
        self._algoritmo = None

    def addAnalizador(self, analizador: Analizador):
        self._analizador = analizador

    def removeAnalizador(self, analizador: Analizador):
        if self._analizador == analizador:
            self._analizador = None

    def exportar_pdf(self) -> str:
        """Genera una representación del reporte en formato PDF."""
        # Lógica de exportación a PDF (simulada)
        print(f"Exportando reporte {self.id} a PDF...")
        return f"reporte_{self.id}.pdf"

    def exportar_json(self) -> str:
        """Genera una representación del reporte en formato JSON."""
        # Lógica de exportación a JSON (simulada)
        print(f"Exportando reporte {self.id} a JSON...")
        return f"reporte_{self.id}.json"