from __future__ import annotations
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .Analizador import Analizador
    from .Algoritmo import Algoritmo
    from .Reporte import Reporte


class Usuario:
    """
    Representa al usuario del sistema.
    Mapea la clase 'Usuario' del diagrama UML.
    """

    def __init__(self, id: int, nombre: str):
        self._id = id
        self._nombre = nombre
        self._analizadores: List[Analizador] = []
        self._algoritmos: List[Algoritmo] = []

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    def nombre(self) -> str:
        return self._nombre

    @nombre.setter
    def nombre(self, value: str):
        self._nombre = value

    @property
    def analizadores(self) -> List[Analizador]:
        return self._analizadores

    @property
    def algoritmos(self) -> List[Algoritmo]:
        return self._algoritmos

    def addAlgoritmo(self, algoritmo: Algoritmo):
        self._algoritmos.append(algoritmo)

    def removeAlgoritmo(self, algoritmo: Algoritmo):
        self._algoritmos.remove(algoritmo)

    def addAnalizador(self, analizador: Analizador):
        self._analizadores.append(analizador)

    def removeAnalizador(self, analizador: Analizador):
        self._analizadores.remove(analizador)

    def ingresar_algoritmo(self, codigo: str) -> Algoritmo:
        """Permite al usuario registrar un nuevo algoritmo."""
        # Lógica para crear y añadir un algoritmo a la lista self._algoritmos
        pass

    def solicitar_analisis(self, algoritmo: Algoritmo) -> Reporte:
        """Inicia el proceso de análisis para un algoritmo dado."""
        # Lógica para usar un analizador y generar un reporte
        pass