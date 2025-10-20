from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Any
import ast  # Importamos el módulo ast para trabajar con los nodos

# Importamos la clase separada que hace el trabajo pesado
from .Algoritmo import Algoritmo
from .Complejidad import Complejidad
from Servicios.EfficiencyVisitor import EfficiencyVisitor
try:
    import sympy
except ImportError:
    sympy = None


if TYPE_CHECKING:
    from .Reporte import Reporte
    from .Parser import Parser
    from Servicios.LLMService import LLMService
    from .Usuario import Usuario
    from Servicios.Ast import AST


class Analizador:
    """
    Clase central que orquesta el análisis de complejidad.
    Mapea la clase 'Analizador' del diagrama UML y ahora contiene la lógica
    de análisis algorítmico.
    """

    def __init__(self, id:int, parser: Parser, llm_service: LLMService):
        self._id = id
        self._parser = parser
        self._llm_service = llm_service
        self._algoritmos: List[Algoritmo] = []
        self._reporte: Reporte | None = None
        self._complejidad: Complejidad | None = None
        self._usuario: Usuario | None = None
        # Atributos para guardar los resultados del último análisis
        self._ultimo_analisis: Dict[str, Any] = {}

    # --- Propiedades existentes (sin cambios) ---
    @property
    def id(self) -> int:
        return self._id

    @id.setter
    def id(self, value: int):
        self._id = value

    @property
    def algoritmos(self) -> List[Algoritmo]:
        return self._algoritmos

    @property
    def reporte(self) -> Reporte | None:
        return self._reporte

    @reporte.setter
    def reporte(self, value: Reporte | None):
        self._reporte = value

    @property
    def complejidad(self) -> Complejidad | None:
        return self._complejidad

    @complejidad.setter
    def complejidad(self, value: Complejidad | None):
        self._complejidad = value

    @property
    def parser(self) -> Parser:
        return self._parser

    @parser.setter
    def parser(self, value: Parser):
        self._parser = value

    @property
    def llm_service(self) -> LLMService:
        return self._llm_service

    @llm_service.setter
    def llm_service(self, value: LLMService):
        self._llm_service = value

    @property
    def usuario(self) -> Usuario | None:
        return self._usuario

    @usuario.setter
    def usuario(self, value: Usuario | None):
        self._usuario = value

    def addAlgoritmo(self, algoritmo: Algoritmo):
        self._algoritmos.append(algoritmo)

    def removeAlgoritmo(self, algoritmo: Algoritmo):
        self._algoritmos.remove(algoritmo)

    def addReporte(self, reporte: Reporte):
        self._reporte = reporte

    # --- LÓGICA DE ANÁLISIS DE COMPLEJIDAD IMPLEMENTADA ---

    def _analizar_ast(self, ast_obj: AST) -> Dict[str, Any]:
        """
        Método central y privado que inspecciona el AST y determina las
        características estructurales para el análisis de complejidad.
        """
        max_profundidad = 0
        hay_salida_temprana = False

        # ast.walk nos permite recorrer todos los nodos del árbol
        for node in ast.walk(ast_obj._arbol):
            # Buscamos bucles (For, While)
            if isinstance(node, (ast.For, ast.While)):
                profundidad_actual = self._calcular_profundidad_anidacion(node)
                if profundidad_actual > max_profundidad:
                    max_profundidad = profundidad_actual

            # Buscamos condiciones de salida temprana (break, return dentro de bucles)
            if isinstance(node, (ast.For, ast.While)):
                for sub_node in ast.walk(node):
                    if isinstance(sub_node, (ast.Break, ast.Return)):
                        hay_salida_temprana = True
                        break
            if hay_salida_temprana:
                break

        return {
            "max_profundidad": max_profundidad,
            "hay_salida_temprana": hay_salida_temprana
        }

    def _calcular_profundidad_anidacion(self, nodo_bucle: ast.AST, profundidad_inicial=1) -> int:
        """
        Calcula la profundidad de los bucles anidados a partir de un nodo inicial.
        """
        max_profundidad_hijo = 0
        # Recorremos solo los hijos directos del cuerpo del bucle
        for hijo in ast.iter_child_nodes(nodo_bucle):
            if isinstance(hijo, (ast.For, ast.While)):
                # Si encontramos un bucle anidado, llamamos recursivamente
                profundidad_hijo = self._calcular_profundidad_anidacion(hijo, profundidad_inicial + 1)
                if profundidad_hijo > max_profundidad_hijo:
                    max_profundidad_hijo = profundidad_hijo

        return max(profundidad_inicial, max_profundidad_hijo)

    def calcular_o(self) -> str:
        """Calcula la cota superior asintótica (Peor Caso)."""
        profundidad = self._ultimo_analisis.get("max_profundidad", 0)
        if profundidad == 0:
            return "O(1)"
        elif profundidad == 1:
            return "O(n)"
        else:
            return f"O(n^{profundidad})"

    def calcular_omega(self) -> str:
        """Calcula la cota inferior asintótica (Mejor Caso)."""
        profundidad = self._ultimo_analisis.get("max_profundidad", 0)
        salida_temprana = self._ultimo_analisis.get("hay_salida_temprana", False)

        if profundidad == 0:
            return "Ω(1)"
        # Si hay una salida temprana (ej. 'break'), el mejor caso podría ser constante.
        if salida_temprana:
            return "Ω(1)"
        # Si no hay salida temprana, el mejor caso es igual al peor.
        elif profundidad == 1:
            return "Ω(n)"
        else:
            return f"Ω(n^{profundidad})"

    def calcular_theta(self) -> str:
        """Calcula la cota ajustada asintótica (Caso Promedio)."""
        o_grande = self.calcular_o()
        omega_grande = self.calcular_omega()

        # Si las cotas superior e inferior coinciden, tenemos una cota ajustada (Theta).
        if o_grande.replace('O', '') == omega_grande.replace('Ω', ''):
            return o_grande.replace('O', 'Θ')
        else:
            # Si no, el análisis promedio es más complejo y a menudo se alinea con el peor caso.
            return f"No se puede determinar una cota Θ simple. El caso promedio tiende a {o_grande}."

    def generar_justificacion(self) -> str:
        """Genera la justificación matemática del análisis basado en la estructura del AST."""
        profundidad = self._ultimo_analisis.get("max_profundidad", 0)
        salida_temprana = self._ultimo_analisis.get("hay_salida_temprana", False)

        if profundidad == 0:
            return "El algoritmo no contiene bucles iterativos ni llamadas recursivas, por lo que su tiempo de ejecución es constante e independiente del tamaño de la entrada."

        justificacion = f"El análisis se basa en la estructura de bucles del algoritmo (Capítulo 2, Introduction to Algorithms).\n"

        if profundidad == 1:
            justificacion += f"- **Peor Caso ({self.calcular_o()})**: Se ha identificado un bucle principal que itera sobre los elementos de la entrada. El tiempo de ejecución crece linealmente con el tamaño 'n' de la entrada.\n"
        else:
            justificacion += f"- **Peor Caso ({self.calcular_o()})**: Se ha detectado una estructura de bucles anidados con una profundidad máxima de {profundidad}. Esto resulta en una complejidad polinómica, ya que por cada elemento del bucle exterior, el interior se ejecuta 'n' veces.\n"

        if salida_temprana:
            justificacion += f"- **Mejor Caso ({self.calcular_omega()})**: El algoritmo contiene una condición de salida temprana (ej. 'break' o 'return' dentro de un bucle). En el mejor de los casos, esta condición se cumple en la primera iteración, resultando en un tiempo de ejecución constante.\n"
        else:
            justificacion += f"- **Mejor Caso ({self.calcular_omega()})**: No se han detectado condiciones de salida temprana. Por lo tanto, el algoritmo debe recorrer la totalidad de la estructura de bucles incluso en el mejor de los casos, igualando la complejidad del peor caso.\n"

        justificacion += f"- **Caso Promedio ({self.calcular_theta()})**: {'Dado que las cotas del mejor y peor caso coinciden, la complejidad promedio es ajustada.' if self.calcular_o().replace('O', '') == self.calcular_omega().replace('Ω', '') else 'La complejidad promedio es más difícil de determinar, pero tiende a seguir el comportamiento del peor caso en la mayoría de las distribuciones de entrada.'}"

        return justificacion

    def _analizar_eficiencia(self, ast_obj: AST) -> Dict[str, Any]:
        """
        Instancia y ejecuta el EfficiencyVisitor para obtener los resultados
        del análisis matemático.
        """
        visitor = EfficiencyVisitor()
        visitor.visit(ast_obj._arbol)

        # Resuelve las sumatorias para obtener las funciones T(n) finales
        t_n_peor = visitor.worst_case_cost.doit()
        t_n_mejor = visitor.best_case_cost.doit()

        return {
            "desglose_costos": visitor.line_costs,
            "funcion_peor_caso": t_n_peor,
            "funcion_mejor_caso": t_n_mejor,
            "funcion_peor_caso_str": str(t_n_peor),
            "funcion_mejor_caso_str": str(t_n_mejor)
        }

    def analizar(self, algoritmo: Algoritmo) -> Complejidad:
        """
        Realiza el análisis completo de un algoritmo, orquestando la creación
        del AST y el análisis de eficiencia para producir un reporte de complejidad.
        """
        if not algoritmo.arbol_sintactico:
            raise ValueError("El algoritmo no tiene un AST. Ejecute el parser primero.")

        # 1. Realizar el análisis de eficiencia detallado
        self._ultimo_analisis = self._analizar_eficiencia(algoritmo.arbol_sintactico)

        t_n_peor = self._ultimo_analisis["funcion_peor_caso"]
        t_n_mejor = self._ultimo_analisis["funcion_mejor_caso"]
        n = sympy.Symbol('n')

        # 2. Derivar O(n) del peor caso y Ω(n) del mejor caso
        orden_peor = sympy.O(t_n_peor, (n, sympy.oo)).args[0]
        orden_mejor = sympy.O(t_n_mejor, (n, sympy.oo)).args[0]

        notacion_o = f"O({orden_peor})"
        notacion_omega = f"Ω({orden_mejor})"

        # 3. Calcular Theta(Θ) si las cotas coinciden
        notacion_theta = f"Θ({orden_peor})" if orden_peor == orden_mejor else "No aplicable"

        # 4. Generar la justificación matemática y detallada
        justificacion = (
            f"El análisis de eficiencia se ha realizado línea por línea, generando funciones de coste para el peor y mejor caso, "
            f"basado en los principios del Capítulo 2 de 'Introduction to Algorithms'.\n\n"
            f"Función de Peor Caso T(n) = {self._ultimo_analisis['funcion_peor_caso_str']}\n"
            f"Función de Mejor Caso T(n) = {self._ultimo_analisis['funcion_mejor_caso_str']}\n\n"
            f"**Desglose de Costos por Línea:**\n"
        )
        for linea, costo, desc in self._ultimo_analisis['desglose_costos']:
            justificacion += f"- Línea {linea}: {desc} | Costo -> {costo}\n"

        justificacion += f"\n**Conclusión Asintótica:**\n"
        justificacion += f"- **Peor Caso (O)**: El término dominante de la función de peor caso es **{orden_peor}**, resultando en una complejidad de **{notacion_o}**.\n"
        justificacion += f"- **Mejor Caso (Ω)**: El término dominante de la función de mejor caso es **{orden_mejor}**, resultando en una complejidad de **{notacion_omega}**.\n"
        justificacion += f"- **Caso Promedio (Θ)**: {f'Dado que las cotas del mejor y peor caso coinciden, la complejidad es **{notacion_theta}**.' if notacion_theta != 'No aplicable' else 'Las cotas del mejor y peor caso difieren, por lo que no se establece una cota ajustada Θ simple.'}"

        # 5. Crear y devolver el objeto Complejidad
        self.complejidad = Complejidad(
            id=algoritmo.id,
            notacion_o=notacion_o,
            notacion_omega=notacion_omega,
            notacion_theta=notacion_theta,
            justificacion=justificacion
        )
        return self.complejidad