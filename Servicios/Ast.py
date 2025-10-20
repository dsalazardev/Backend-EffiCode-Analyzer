from __future__ import annotations
import ast
from typing import List, Dict, Any


class AST:
    """
    Un wrapper alrededor del módulo 'ast' de la librería estándar de Python.

    Esta clase toma una cadena de código fuente de Python, la parsea en un
    Árbol de Sintaxis Abstracta (AST), y proporciona métodos de alto nivel
    para analizar la estructura del código. Sirve como puente entre el
    código (potencialmente traducido desde pseudocódigo por un LLM) y la
    lógica de análisis de complejidad.
    """

    def __init__(self, codigo: str):
        """
        Inicializa y parsea el código fuente para construir el AST.

        Args:
            codigo (str): Una cadena que contiene código Python válido.

        Raises:
            SyntaxError: Si el código fuente no es sintácticamente correcto.
        """
        self._codigo = codigo
        self._arbol: ast.Module = ast.parse(codigo)

    def extraer_funciones(self) -> List[str]:
        """
        Recorre el AST y devuelve los nombres de todas las funciones definidas.

        Returns:
            List[str]: Una lista con los nombres de las funciones (def).
        """
        return [
            nodo.name
            for nodo in ast.walk(self._arbol)
            if isinstance(nodo, ast.FunctionDef)
        ]

    def extraer_bucles(self) -> List[str]:
        """
        Recorre el AST y devuelve los tipos de bucles encontrados.

        Returns:
            List[str]: Una lista de cadenas, como ['For', 'While'].
        """
        tipos_bucles = []
        for nodo in ast.walk(self._arbol):
            if isinstance(nodo, ast.For):
                tipos_bucles.append('For')
            elif isinstance(nodo, ast.While):
                tipos_bucles.append('While')
        return tipos_bucles

    def extraer_condicionales(self) -> int:
        """
        Cuenta el número total de sentencias 'if' en el AST.

        Nota: Esto no incluye expresiones ternarias (if x else y).

        Returns:
            int: El número de nodos 'if' (ast.If).
        """
        return sum(1 for nodo in ast.walk(self._arbol) if isinstance(nodo, ast.If))

    def extraer_llamadas(self) -> List[str]:
        """
        Recorre el AST y devuelve los nombres de todas las funciones que se llaman.

        Returns:
            List[str]: Una lista con los nombres de las funciones en nodos Call.
        """
        llamadas = []
        for nodo in ast.walk(self._arbol):
            if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
                llamadas.append(nodo.func.id)
        return sorted(list(set(llamadas)))  # Devolver únicos y ordenados

    def contar_nodos(self) -> int:
        """
        Calcula el número total de nodos en el AST.

        Returns:
            int: La cantidad total de nodos.
        """
        return sum(1 for _ in ast.walk(self._arbol))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convierte el AST completo a una estructura de diccionario anidado.

        Esto es útil para depuración, serialización a JSON, o para un análisis
        más detallado de la estructura del árbol.

        Returns:
            Dict[str, Any]: Una representación del AST en formato de diccionario.
        """
        return self._ast_a_dict(self._arbol)

    def _ast_a_dict(self, nodo: ast.AST) -> Dict[str, Any] | List[Any] | Any:
        """
        Función auxiliar recursiva para convertir un nodo del AST a diccionario.
        """
        if isinstance(nodo, ast.AST):
            # Obtiene el nombre de la clase del nodo (ej. 'Module', 'FunctionDef')
            resultado = {'_type': nodo.__class__.__name__}
            # Itera sobre los campos definidos para ese tipo de nodo
            for campo, valor in ast.iter_fields(nodo):
                resultado[campo] = self._ast_a_dict(valor)
            return resultado
        elif isinstance(nodo, list):
            return [self._ast_a_dict(item) for item in nodo]
        # Devuelve los tipos primitivos tal cual
        return nodo


# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Código de ejemplo: un algoritmo de búsqueda lineal
    codigo_python = """
def busqueda_lineal(lista, objetivo):
    encontrado = False
    for i in range(len(lista)):
        if lista[i] == objetivo:
            encontrado = True
            break

    if encontrado:
        print(f"Elemento {objetivo} encontrado.")
    else:
        print(f"Elemento {objetivo} no encontrado.")

    return encontrado
"""

    print("--- Analizando el siguiente código ---")
    print(codigo_python)

    # 1. Instanciar la clase AST con el código
    try:
        analizador_ast = AST(codigo_python)

        # 2. Usar los métodos de análisis
        print(f"Número total de nodos en el AST: {analizador_ast.contar_nodos()}")
        print("-" * 20)

        funciones = analizador_ast.extraer_funciones()
        print(f"Funciones definidas: {funciones}")

        bucles = analizador_ast.extraer_bucles()
        print(f"Tipos de bucles encontrados: {bucles}")

        num_ifs = analizador_ast.extraer_condicionales()
        print(f"Número de condicionales (if): {num_ifs}")

        llamadas = analizador_ast.extraer_llamadas()
        print(f"Funciones llamadas: {llamadas}")
        print("-" * 20)

        # 3. Exportar a diccionario (imprimimos solo una parte para brevedad)
        # import json
        # ast_dict = analizador_ast.to_dict()
        # print("AST exportado a JSON (primer nivel):")
        # print(json.dumps(ast_dict, indent=2))

    except SyntaxError as e:
        print(f"Error de sintaxis en el código: {e}")