from __future__ import annotations
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .Analizador import Analizador
    from .Reporte import Reporte

class Complejidad:
    """
    Almacena el resultado del análisis de complejidad.
    Mapea la clase 'Complejidad' del diagrama UML.
    """
    def __init__(self, id: int, notacion_o: str, notacion_omega: str, notacion_theta: str, justificacion: str, justification_data: dict = None):
        self._id = id
        self._notacion_o = notacion_o
        self._notacion_omega = notacion_omega
        self._notacion_theta = notacion_theta
        self._justificacion_matematica = justificacion
        self._justification_data = justification_data or {}
        self._analizador: Optional[Analizador] = None
        self._reporte: Optional[Reporte] = None

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    def notacion_o(self) -> str:
        return self._notacion_o

    @notacion_o.setter
    def notacion_o(self, value: str):
        self._notacion_o = value

    @property
    def notacion_omega(self) -> str:
        return self._notacion_omega

    @notacion_omega.setter
    def notacion_omega(self, value: str):
        self._notacion_omega = value

    @property
    def notacion_theta(self) -> str:
        return self._notacion_theta

    @notacion_theta.setter
    def notacion_theta(self, value: str):
        self._notacion_theta = value

    @property
    def justificacion_matematica(self) -> str:
        return self._justificacion_matematica

    @justificacion_matematica.setter
    def justificacion_matematica(self, value: str):
        self._justificacion_matematica = value

    @property
    def justification_data(self) -> dict:
        return self._justification_data

    @justification_data.setter
    def justification_data(self, value: dict):
        self._justification_data = value

    @property
    def analizador(self) -> Optional[Analizador]:
        return self._analizador

    @analizador.setter
    def analizador(self, value: Optional[Analizador]):
        self._analizador = value

    @property
    def reporte(self) -> Optional[Reporte]:
        return self._reporte

    @reporte.setter
    def reporte(self, value: Optional[Reporte]):
        self._reporte = value

    def addAnalizador(self, analizador: Analizador):
        self._analizador = analizador

    def removeAnalizador(self, analizador: Analizador):
        if self._analizador == analizador:
            self._analizador = None