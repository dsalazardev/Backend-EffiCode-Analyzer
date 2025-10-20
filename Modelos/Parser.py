from __future__ import annotations
from typing import TYPE_CHECKING

from Servicios.Grammar import Grammar
from Servicios.LLMService import LLMService
from Servicios.Ast import AST

# Importaciones de tipos para evitar dependencias circulares
if TYPE_CHECKING:
    from .Analizador import Analizador

class Parser:

    def __init__(self, id: int, gramatica: Grammar, llm_service: LLMService):
        self._id = id
        self._gramatica = gramatica
        self._llm_service = llm_service
        self._analizador: Analizador | None = None

    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    def gramatica(self) -> Grammar:
        return self._gramatica

    @gramatica.setter
    def gramatica(self, value: Grammar):
        self._gramatica = value

    @property
    def analizador(self) -> Analizador | None:
        return self._analizador

    @analizador.setter
    def analizador(self, value: Analizador | None):
        self._analizador = value

    def addAnalizador(self, analizador: Analizador):
        self._analizador = analizador

    def removeAnalizador(self, analizador: Analizador):
        if self._analizador == analizador:
            self._analizador = None

    def validar_sintaxis(self, pseudocodigo: str) -> bool:
        """
        Verifica si el código cumple con la gramática definida.
        Delega la validación al servicio de Gramática.
        """
        return self._gramatica.validar_sentencia(pseudocodigo)

    def parsear(self, pseudocodigo: str) -> AST:
        """
        Convierte el pseudocódigo en un Árbol de Sintaxis Abstracta (AST) de Python.
        El proceso implica validar la sintaxis del pseudocódigo, traducirlo a
        Python usando un LLM y finalmente, parsear ese código Python a un AST.

        Args:
            pseudocodigo (str): El código fuente en pseudocódigo estilo Cormen.

        Returns:
            AST: Una instancia del servicio AST que contiene el árbol del código Python.

        Raises:
            SyntaxError: Si la sintaxis del pseudocódigo es inválida o si el
                         código Python generado por el LLM es inválido.
            ConnectionError: Si hay un problema con el servicio de traducción del LLM.
        """
        # 1. Validar la sintaxis usando el servicio de Gramática
        if not self.validar_sintaxis(pseudocodigo):
            raise SyntaxError("Error de sintaxis: El pseudocódigo no es válido según la gramática.")

        # 2. Traducir a Python usando el servicio LLM
        codigo_python = self._llm_service.traducir_pseudocodigo_a_python(pseudocodigo)
        if not codigo_python or "# Error" in codigo_python:
            raise ConnectionError(f"Error de traducción: El servicio LLM falló. Detalles: {codigo_python}")

        # 3. Parsear el código Python y devolver el objeto AST
        try:
            ast_obj = AST(codigo_python)
            return ast_obj
        except SyntaxError as e:
            # Este error ocurre si el LLM devuelve código Python que no es sintácticamente correcto
            raise SyntaxError(
                f"El código generado por el LLM no es válido. Error: {e}\nCódigo recibido:\n{codigo_python}")
