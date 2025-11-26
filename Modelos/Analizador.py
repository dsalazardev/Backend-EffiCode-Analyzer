from __future__ import annotations
from typing import TYPE_CHECKING, List, Dict, Any, Tuple
import ast
import json

from .Algoritmo import Algoritmo
from .Complejidad import Complejidad
from Servicios.EfficiencyVisitor import EfficiencyVisitor

try:
    import sympy
    from sympy import Symbol, Sum, simplify, expand, latex, oo, Rational
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
    Orquesta el análisis de complejidad a partir del AST generado por el parser.
    Incluye resolución paso a paso de sumatorias para justificación matemática.
    Soporta tanto algoritmos iterativos como recursivos.
    """

    def __init__(self, id: int, parser: 'Parser', llm_service: 'LLMService'):
        self._id = id
        self._parser = parser
        self._llm_service = llm_service
        self._algoritmos: List[Algoritmo] = []
        self._reporte: 'Reporte' | None = None
        self._complejidad: Complejidad | None = None
        self._usuario: 'Usuario' | None = None
        self._ultimo_analisis: Dict[str, Any] = {}

    @property
    def id(self) -> int:
        return self._id

    @property
    def parser(self) -> 'Parser':
        return self._parser

    @parser.setter
    def parser(self, value: 'Parser'):
        self._parser = value

    @property
    def llm_service(self) -> 'LLMService':
        return self._llm_service

    @llm_service.setter
    def llm_service(self, value: 'LLMService'):
        self._llm_service = value

    def addAlgoritmo(self, algoritmo: Algoritmo):
        self._algoritmos.append(algoritmo)

    def removeAlgoritmo(self, algoritmo: Algoritmo):
        self._algoritmos.remove(algoritmo)

    def _sanitize_latex(self, tex: str) -> str:
        """Limpia la salida LaTeX de sympy para compatibilidad con KaTeX."""
        try:
            if not isinstance(tex, str):
                tex = str(tex)
            tex = tex.replace('\\left', '').replace('\\right', '')
            tex = ' '.join(tex.split())
            return tex
        except Exception:
            return str(tex)

    # ==========================================
    # SECCIÓN: ANÁLISIS DE ALGORITMOS ITERATIVOS
    # ==========================================

    def _generar_pasos_peor_caso(self, max_profundidad: int) -> List[Dict[str, str]]:
        """
        Genera los pasos de resolución matemática para el PEOR CASO.
        Usa sumatorias estándar del libro de Cormen.
        """
        if not sympy:
            return []
        
        steps = []
        
        if max_profundidad == 0:
            steps.append({
                'step': 1,
                'title': 'Identificación del algoritmo',
                'description': 'El algoritmo no contiene bucles iterativos.',
                'latex': 'T(n) = c_1',
                'explanation': 'Solo operaciones de tiempo constante.'
            })
            steps.append({
                'step': 2,
                'title': 'Resultado final',
                'description': 'Complejidad constante.',
                'latex': 'T(n) = \\Theta(1)',
                'explanation': 'El tiempo de ejecución no depende del tamaño de entrada.'
            })
            
        elif max_profundidad == 1:
            steps.append({
                'step': 1,
                'title': 'Identificación de la estructura',
                'description': 'El algoritmo contiene un bucle simple que itera n veces.',
                'latex': 'T(n) = c_1 + \\sum_{j=1}^{n} c_2',
                'explanation': 'El bucle externo se ejecuta n veces, cada iteración tiene costo c₂.'
            })
            steps.append({
                'step': 2,
                'title': 'Resolver la sumatoria',
                'description': 'Aplicamos la fórmula de suma de constantes.',
                'latex': '\\sum_{j=1}^{n} c_2 = c_2 \\cdot n',
                'explanation': 'La suma de una constante n veces es n por esa constante.'
            })
            steps.append({
                'step': 3,
                'title': 'Expresión simplificada',
                'description': 'Combinamos los términos.',
                'latex': 'T(n) = c_1 + c_2 \\cdot n',
                'explanation': 'Función lineal en n.'
            })
            steps.append({
                'step': 4,
                'title': 'Notación asintótica',
                'description': 'Identificamos el término dominante.',
                'latex': 'T(n) = \\Theta(n)',
                'explanation': 'El término c₂·n domina cuando n → ∞, por lo tanto es O(n).'
            })
            
        else:  # max_profundidad >= 2 (bucles anidados como Insertion Sort)
            steps.append({
                'step': 1,
                'title': 'Función de tiempo T(n) - Peor Caso',
                'description': 'Sumamos los costos de cada línea del algoritmo. En el peor caso (array en orden inverso), el bucle while se ejecuta j-1 veces para cada j.',
                'latex': 'T(n) = c_1 \\cdot n + c_2(n-1) + c_4(n-1) + c_3\\sum_{j=2}^{n} j + c_5\\sum_{j=2}^{n}(j-1) + c_6\\sum_{j=2}^{n}(j-1) + c_7(n-1)',
                'explanation': 'Cada línea contribuye: bucle for (c₁·n), asignaciones (c₂, c₄, c₇ ejecutadas n-1 veces), y el while con sus operaciones internas que dependen de j.'
            })
            steps.append({
                'step': 2,
                'title': 'Resolver sumatoria del while (iteraciones)',
                'description': 'Calculamos cuántas veces se ejecuta el bucle while en total.',
                'latex': '\\sum_{j=2}^{n} j = \\frac{n(n+1)}{2} - 1 = \\frac{n^2 + n - 2}{2}',
                'explanation': 'Usamos la fórmula de suma aritmética: Σj = n(n+1)/2, restamos 1 porque empezamos en j=2.'
            })
            steps.append({
                'step': 3,
                'title': 'Resolver sumatoria del cuerpo del while',
                'description': 'Las operaciones dentro del while se ejecutan (j-1) veces por cada j.',
                'latex': '\\sum_{j=2}^{n} (j-1) = \\sum_{k=1}^{n-1} k = \\frac{(n-1)n}{2} = \\frac{n^2 - n}{2}',
                'explanation': 'Sustitución k = j-1. Aplicamos fórmula de suma: Σk = (n-1)n/2.'
            })
            steps.append({
                'step': 4,
                'title': 'Sustituir las sumatorias resueltas',
                'description': 'Reemplazamos las sumatorias por sus valores calculados.',
                'latex': 'T(n) = c_1 n + c_2(n-1) + c_4(n-1) + c_3\\frac{n^2+n-2}{2} + (c_5+c_6)\\frac{n^2-n}{2} + c_7(n-1)',
                'explanation': 'Sustituimos los resultados de los pasos 2 y 3.'
            })
            steps.append({
                'step': 5,
                'title': 'Expandir y agrupar términos',
                'description': 'Expandimos los productos y agrupamos por potencias de n.',
                'latex': 'T(n) = \\left(\\frac{c_3}{2} + \\frac{c_5+c_6}{2}\\right)n^2 + \\left(c_1 + c_2 + c_4 + \\frac{c_3}{2} - \\frac{c_5+c_6}{2} + c_7\\right)n + (\\text{constantes})',
                'explanation': 'Reorganizamos para ver claramente los coeficientes de n², n y términos constantes.'
            })
            steps.append({
                'step': 6,
                'title': 'Forma general cuadrática',
                'description': 'Expresamos T(n) en forma polinomial.',
                'latex': 'T(n) = an^2 + bn + c',
                'explanation': 'Donde a, b, c son constantes positivas. Esta es una función cuadrática.'
            })
            steps.append({
                'step': 7,
                'title': 'Análisis asintótico',
                'description': 'Identificamos el término dominante cuando n → ∞.',
                'latex': '\\lim_{n \\to \\infty} \\frac{T(n)}{n^2} = \\lim_{n \\to \\infty} \\frac{an^2 + bn + c}{n^2} = a',
                'explanation': 'El coeficiente a es una constante positiva, confirmando que n² es el término dominante.'
            })
            steps.append({
                'step': 8,
                'title': 'Conclusión Peor Caso',
                'description': 'Determinamos la notación Big-O.',
                'latex': 'T(n) = \\Theta(n^2) \\implies O(n^2)',
                'explanation': 'En el peor caso, Insertion Sort tiene complejidad cuadrática O(n²).'
            })
        
        return steps

    def _generar_pasos_mejor_caso(self, max_profundidad: int, hay_salida_temprana: bool) -> List[Dict[str, str]]:
        """
        Genera los pasos de resolución matemática para el MEJOR CASO.
        """
        if not sympy:
            return []
        
        steps = []
        
        if max_profundidad == 0:
            steps.append({
                'step': 1,
                'title': 'Mejor caso',
                'description': 'Sin bucles, el mejor y peor caso son iguales.',
                'latex': 'T(n) = \\Theta(1)',
                'explanation': 'Complejidad constante en todos los casos.'
            })
            
        elif max_profundidad == 1:
            if hay_salida_temprana:
                steps.append({
                    'step': 1,
                    'title': 'Mejor caso con salida temprana',
                    'description': 'El elemento buscado está al inicio o se cumple la condición inmediatamente.',
                    'latex': 'T(n) = c_1 + c_2 = \\Theta(1)',
                    'explanation': 'El bucle termina en la primera iteración.'
                })
            else:
                steps.append({
                    'step': 1,
                    'title': 'Mejor caso',
                    'description': 'El bucle debe recorrer todos los elementos.',
                    'latex': 'T(n) = \\Theta(n)',
                    'explanation': 'Igual al peor caso para un bucle simple sin salida temprana.'
                })
                
        else:  # Bucles anidados (Insertion Sort)
            steps.append({
                'step': 1,
                'title': 'Función de tiempo T(n) - Mejor Caso',
                'description': 'En el mejor caso, el array ya está ordenado. El bucle while nunca ejecuta su cuerpo porque A[i] ≤ key siempre.',
                'latex': 'T(n) = c_1 \\cdot n + c_2(n-1) + c_4(n-1) + c_3(n-1) + c_7(n-1)',
                'explanation': 'El test del while (c₃) se ejecuta 1 vez por iteración (solo verifica y sale), pero el cuerpo (c₅, c₆) no se ejecuta nunca.'
            })
            steps.append({
                'step': 2,
                'title': 'Simplificar la expresión',
                'description': 'Agrupamos los términos lineales.',
                'latex': 'T(n) = c_1 n + (c_2 + c_3 + c_4 + c_7)(n-1)',
                'explanation': 'Factorizamos (n-1) de los términos que se ejecutan n-1 veces.'
            })
            steps.append({
                'step': 3,
                'title': 'Expandir',
                'description': 'Distribuimos y combinamos.',
                'latex': 'T(n) = c_1 n + (c_2 + c_3 + c_4 + c_7)n - (c_2 + c_3 + c_4 + c_7)',
                'explanation': 'Expandimos el producto.'
            })
            steps.append({
                'step': 4,
                'title': 'Forma lineal',
                'description': 'Expresamos como función lineal.',
                'latex': 'T(n) = (c_1 + c_2 + c_3 + c_4 + c_7)n - (c_2 + c_3 + c_4 + c_7)',
                'explanation': 'T(n) = an + b, donde a y b son constantes.'
            })
            steps.append({
                'step': 5,
                'title': 'Análisis asintótico',
                'description': 'Identificamos el término dominante.',
                'latex': '\\lim_{n \\to \\infty} \\frac{T(n)}{n} = a > 0',
                'explanation': 'El término lineal domina, la constante se vuelve despreciable.'
            })
            steps.append({
                'step': 6,
                'title': 'Conclusión Mejor Caso',
                'description': 'Determinamos la notación Omega.',
                'latex': 'T(n) = \\Theta(n) \\implies \\Omega(n)',
                'explanation': 'En el mejor caso, Insertion Sort tiene complejidad lineal Ω(n).'
            })
        
        return steps

    def _analizar_eficiencia(self, ast_obj: 'AST') -> Dict[str, Any]:
        visitor = EfficiencyVisitor()
        visitor.visit(ast_obj._arbol)
        t_n_peor = visitor.worst_case_cost
        t_n_mejor = visitor.best_case_cost
        return {
            "desglose_costos": visitor.line_costs,
            "funcion_peor_caso": t_n_peor,
            "funcion_mejor_caso": t_n_mejor,
            "funcion_peor_caso_str": str(t_n_peor),
            "funcion_mejor_caso_str": str(t_n_mejor)
        }

    def _analizar_estructura_ast(self, ast_obj: 'AST') -> Dict[str, Any]:
        """
        Analiza la estructura del AST para determinar profundidad de bucles
        y condiciones de salida temprana.
        """
        max_profundidad = 0
        hay_salida_temprana = False
        
        def contar_profundidad(nodo, profundidad_actual=0):
            nonlocal max_profundidad, hay_salida_temprana
            
            if isinstance(nodo, (ast.For, ast.While)):
                profundidad_actual += 1
                if profundidad_actual > max_profundidad:
                    max_profundidad = profundidad_actual
            
            if isinstance(nodo, (ast.Break, ast.Return)):
                hay_salida_temprana = True
            
            for hijo in ast.iter_child_nodes(nodo):
                contar_profundidad(hijo, profundidad_actual)
        
        contar_profundidad(ast_obj._arbol)
        
        return {
            "max_profundidad": max_profundidad,
            "hay_salida_temprana": hay_salida_temprana
        }

    def analizar(self, algoritmo: Algoritmo) -> Complejidad:
        """
        Analiza algoritmos ITERATIVOS con resolución paso a paso.
        """
        if not algoritmo.arbol_sintactico:
            raise ValueError("El algoritmo no tiene un AST. Ejecute el parser primero.")

        # Análisis de eficiencia (visitor)
        self._ultimo_analisis = self._analizar_eficiencia(algoritmo.arbol_sintactico)
        
        # Análisis estructural del AST
        estructura = self._analizar_estructura_ast(algoritmo.arbol_sintactico)
        self._ultimo_analisis.update(estructura)

        t_n_peor = self._ultimo_analisis.get("funcion_peor_caso")
        t_n_mejor = self._ultimo_analisis.get("funcion_mejor_caso")
        max_profundidad = estructura.get("max_profundidad", 0)
        hay_salida_temprana = estructura.get("hay_salida_temprana", False)

        n = sympy.Symbol('n') if sympy else None

        # Determinar complejidades basadas en la estructura de bucles
        if max_profundidad == 0:
            orden_peor_str = "1"
            orden_mejor_str = "1"
        elif max_profundidad == 1:
            orden_peor_str = "n"
            orden_mejor_str = "n" if not hay_salida_temprana else "1"
        else:
            orden_peor_str = f"n^{max_profundidad}"
            orden_mejor_str = "n" if hay_salida_temprana or max_profundidad >= 2 else f"n^{max_profundidad}"

        notacion_o = f"O({orden_peor_str})"
        notacion_omega = f"Ω({orden_mejor_str})"
        
        if orden_peor_str == orden_mejor_str:
            notacion_theta = f"Θ({orden_peor_str})"
        else:
            notacion_theta = "No aplicable"

        # Generar pasos de resolución matemática
        pasos_peor_caso = self._generar_pasos_peor_caso(max_profundidad)
        pasos_mejor_caso = self._generar_pasos_mejor_caso(max_profundidad, hay_salida_temprana)

        # Procesar desglose de costos para LaTeX
        raw_desglose = self._ultimo_analisis.get('desglose_costos', [])
        line_costs = []
        for entry in raw_desglose:
            ln = None
            desc = ''
            cost_latex = ''
            if isinstance(entry, (list, tuple)) and len(entry) == 3:
                ln, cost_str, desc = entry
                cost_latex = self._sanitize_latex(cost_str) if isinstance(cost_str, str) else self._sanitize_latex(str(cost_str))
            elif isinstance(entry, (list, tuple)) and len(entry) == 4:
                ln, worst_expr, best_expr, desc = entry
                try:
                    worst_tex = self._sanitize_latex(sympy.latex(worst_expr)) if sympy else self._sanitize_latex(str(worst_expr))
                except Exception:
                    worst_tex = self._sanitize_latex(str(worst_expr))
                try:
                    best_tex = self._sanitize_latex(sympy.latex(best_expr)) if sympy else self._sanitize_latex(str(best_expr))
                except Exception:
                    best_tex = self._sanitize_latex(str(best_expr))
                if worst_tex == best_tex:
                    cost_latex = worst_tex
                else:
                    cost_latex = f"Peor: {worst_tex}, Mejor: {best_tex}"
            else:
                try:
                    ln = int(entry[0])
                except Exception:
                    ln = None
                desc = str(entry[-1]) if entry else ''
                cost_latex = str(entry[1]) if len(entry) > 1 else ''

            line_costs.append({
                'line': ln,
                'description': desc,
                'cost': cost_latex
            })

        # Generar LaTeX para las funciones T(n)
        try:
            worst_case_func_str = self._sanitize_latex(sympy.latex(t_n_peor)) if sympy else self._sanitize_latex(str(t_n_peor))
        except Exception:
            worst_case_func_str = self._sanitize_latex(str(t_n_peor))
        try:
            best_case_func_str = self._sanitize_latex(sympy.latex(t_n_mejor)) if sympy else self._sanitize_latex(str(t_n_mejor))
        except Exception:
            best_case_func_str = self._sanitize_latex(str(t_n_mejor))

        dominant_worst_tex = orden_peor_str
        dominant_best_tex = orden_mejor_str

        # Generar justificación basada en la estructura
        if max_profundidad == 0:
            justificacion = "El algoritmo no contiene bucles iterativos, por lo que su tiempo de ejecución es constante O(1)."
        elif max_profundidad == 1:
            justificacion = f"El algoritmo contiene un bucle simple que itera sobre los elementos de entrada, resultando en complejidad lineal {notacion_o}."
        else:
            justificacion = (
                f"El algoritmo contiene bucles anidados con profundidad {max_profundidad}. "
                f"En el peor caso (ej. array en orden inverso), el bucle interno se ejecuta O(n) veces "
                f"por cada iteración del bucle externo, resultando en {notacion_o}. "
                f"En el mejor caso (ej. array ya ordenado), el bucle interno puede ejecutarse "
                f"en tiempo constante por iteración, resultando en {notacion_omega}."
            )

        justification_data = {
            'worst_case_function': worst_case_func_str,
            'best_case_function': best_case_func_str,
            'line_costs': line_costs,
            'resolution_steps': {
                'worst_case': pasos_peor_caso,
                'best_case': pasos_mejor_caso
            },
            'conclusion': {
                'worst_case': {'dominant_term': dominant_worst_tex, 'complexity': notacion_o},
                'best_case': {'dominant_term': dominant_best_tex, 'complexity': notacion_omega},
                'average_case': {'complexity': notacion_theta, 'description': f"El caso promedio tiende a {notacion_o}" if notacion_theta == "No aplicable" else notacion_theta}
            }
        }

        complejidad = Complejidad(
            self._id,
            notacion_o,
            notacion_omega,
            notacion_theta,
            justificacion,
            justification_data
        )

        complejidad.analizador = self
        self._complejidad = complejidad
        return complejidad

    # ==========================================
    # SECCIÓN: ANÁLISIS DE ALGORITMOS RECURSIVOS
    # ==========================================

    def _detectar_recursividad(self, ast_obj: 'AST') -> bool:
        """
        Detecta si la función principal realiza una llamada recursiva a sí misma.
        """
        funciones_definidas = ast_obj.extraer_funciones()
        if not funciones_definidas:
            return False

        nombre_funcion_principal = funciones_definidas[0]
        llamadas = ast_obj.extraer_llamadas()
        
        return nombre_funcion_principal in llamadas

    def _detectar_patron_recursivo(self, ast_obj: 'AST') -> Dict[str, Any]:
        """
        Detecta el patrón recursivo específico del algoritmo.
        """
        funciones_definidas = ast_obj.extraer_funciones()
        nombre_funcion_principal = funciones_definidas[0] if funciones_definidas else ""
        
        parametros_recursion = self._analizar_parametros_recursivos(ast_obj, nombre_funcion_principal)
        tipo_recursion = self._clasificar_tipo_recursion(ast_obj, nombre_funcion_principal)
        division_info = self._analizar_division_problema(ast_obj, nombre_funcion_principal)
        
        return {
            "tipo": tipo_recursion,
            "parametros": parametros_recursion,
            "division": division_info,
            "nombre_funcion": nombre_funcion_principal
        }

    def _analizar_parametros_recursivos(self, ast_obj: 'AST', nombre_funcion: str) -> Dict[str, Any]:
        """
        Analiza cómo cambian los parámetros en las llamadas recursivas.
        """
        llamadas_recursivas = []
        
        for node in ast.walk(ast_obj._arbol):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == nombre_funcion:
                    args_analysis = []
                    for arg in node.args:
                        if isinstance(arg, ast.BinOp):
                            args_analysis.append(self._analizar_expresion_recursiva(arg))
                        elif isinstance(arg, ast.Constant):
                            args_analysis.append({"tipo": "constante", "valor": arg.value})
                        else:
                            args_analysis.append({"tipo": "variable", "expresion": ast.unparse(arg)})
                    
                    llamadas_recursivas.append({
                        "nodo": node,
                        "argumentos": args_analysis,
                        "linea": node.lineno
                    })
        
        return {
            "llamadas": llamadas_recursivas,
            "total_llamadas": len(llamadas_recursivas)
        }

    def _analizar_expresion_recursiva(self, node: ast.BinOp) -> Dict[str, Any]:
        """
        Analiza expresiones binarias en argumentos recursivos (n-1, n//2, etc.)
        """
        if isinstance(node, ast.BinOp):
            izquierda = ast.unparse(node.left) if hasattr(node, 'left') else "?"
            derecha = ast.unparse(node.right) if hasattr(node, 'right') else "?"
            operador = type(node.op).__name__
            
            op_map = {
                'Sub': '-',
                'Add': '+',
                'Mult': '*',
                'Div': '/',
                'FloorDiv': '//'
            }
            
            return {
                "tipo": "operacion_binaria",
                "expresion": f"{izquierda}{op_map.get(operador, operador)}{derecha}",
                "operador": operador,
                "operandos": [izquierda, derecha]
            }
        
        return {"tipo": "desconocido", "expresion": ast.unparse(node)}

    def _clasificar_tipo_recursion(self, ast_obj: 'AST', nombre_funcion: str) -> str:
        """
        Clasifica el tipo de recursión: lineal, binaria, múltiple, etc.
        """
        llamadas_recursivas = []
        for node in ast.walk(ast_obj._arbol):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == nombre_funcion:
                    llamadas_recursivas.append(node)
        
        if not llamadas_recursivas:
            return "no_recursiva"
        
        llamadas_por_camino = self._contar_llamadas_por_camino(ast_obj, nombre_funcion)
        max_llamadas_en_camino = max(llamadas_por_camino.values()) if llamadas_por_camino else 0
        
        if max_llamadas_en_camino == 2:
            if self._es_patron_fibonacci(ast_obj, nombre_funcion):
                return "recursion_exponencial_fibonacci"
            else:
                return "recursion_binaria"
        
        elif max_llamadas_en_camino == 1:
            if len(llamadas_por_camino) > 1:
                return "recursion_multiple"
            else:
                return "recursion_lineal"
        
        elif max_llamadas_en_camino > 2:
            return "recursion_multiple_compleja"
        
        return "recursion_general"

    def _es_patron_fibonacci(self, ast_obj: 'AST', nombre_funcion: str) -> bool:
        """
        Detecta si el algoritmo sigue el patrón Fibonacci: F(n) = F(n-1) + F(n-2)
        """
        llamadas_n_minus_1 = 0
        llamadas_n_minus_2 = 0
        
        for node in ast.walk(ast_obj._arbol):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == nombre_funcion:
                    for arg in node.args:
                        arg_str = ast.unparse(arg).replace(" ", "")
                        if "n-1" in arg_str or "-1" in arg_str:
                            llamadas_n_minus_1 += 1
                        elif "n-2" in arg_str or "-2" in arg_str:
                            llamadas_n_minus_2 += 1
        
        return llamadas_n_minus_1 >= 1 and llamadas_n_minus_2 >= 1

    def _contar_llamadas_por_camino(self, ast_obj: 'AST', nombre_funcion: str) -> Dict[str, int]:
        """
        Cuenta las llamadas recursivas en cada camino de ejecución (if/else branches).
        """
        contador = {}
        
        def contar_en_nodo(nodo, camino_actual="main"):
            if isinstance(nodo, list):
                for elemento in nodo:
                    contar_en_nodo(elemento, camino_actual)
                return
            
            if isinstance(nodo, ast.Call) and isinstance(nodo.func, ast.Name):
                if nodo.func.id == nombre_funcion:
                    contador[camino_actual] = contador.get(camino_actual, 0) + 1
            
            if isinstance(nodo, ast.If):
                contar_en_nodo(nodo.body, f"{camino_actual}_if")
                if nodo.orelse:
                    contar_en_nodo(nodo.orelse, f"{camino_actual}_else")
            else:
                for nombre_campo, valor in ast.iter_fields(nodo):
                    if isinstance(valor, (ast.AST, list)):
                        contar_en_nodo(valor, camino_actual)
        
        funcion_def = next((n for n in ast.walk(ast_obj._arbol) if isinstance(n, ast.FunctionDef)), None)
        if funcion_def:
            contar_en_nodo(funcion_def.body)
        
        return contador

    def _analizar_division_problema(self, ast_obj: 'AST', nombre_funcion: str) -> Dict[str, Any]:
        """
        Analiza cómo se divide el problema en subproblemas.
        """
        patrones = {
            "mitad": ["//2", "/2", ">>1", "mid", "middle"],
            "tercio": ["//3", "/3"],
            "n_minus_1": ["-1", "n-1", "size-1"],
            "n_minus_k": ["-", "sub", "minus"]
        }
        
        division_detectada = "desconocida"
        factor_division = 2
        
        for node in ast.walk(ast_obj._arbol):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == nombre_funcion:
                    for arg in node.args:
                        arg_str = ast.unparse(arg).lower()
                        
                        if any(patron in arg_str for patron in patrones["mitad"]):
                            division_detectada = "mitad"
                            factor_division = 2
                        elif any(patron in arg_str for patron in patrones["tercio"]):
                            division_detectada = "tercio"
                            factor_division = 3
                        elif any(patron in arg_str for patron in patrones["n_minus_1"]):
                            division_detectada = "n_minus_1"
                            factor_division = 1
                        elif any(patron in arg_str for patron in patrones["n_minus_k"]):
                            division_detectada = "n_minus_k"
                            factor_division = "variable"
        
        return {
            "tipo": division_detectada,
            "factor": factor_division
        }

    def _generar_ecuacion_recurrencia_mejorada(self, ast_obj: 'AST') -> Dict[str, Any]:
        """
        Genera la ecuación de recurrencia basada en el análisis estructural.
        """
        patron_recursivo = self._detectar_patron_recursivo(ast_obj)
        
        a = 0
        b = 2
        f_n = "1"
        tipo_especial = None
        
        if patron_recursivo["tipo"] == "recursion_lineal":
            a = 1
            b = 1
            tipo_especial = "n_minus_1"
            
        elif patron_recursivo["tipo"] == "recursion_binaria":
            a = 2
            
        elif patron_recursivo["tipo"] == "recursion_multiple":
            a = 1
            
        elif patron_recursivo["tipo"] == "recursion_exponencial_fibonacci":
            a = 2
            b = 1
            tipo_especial = "fibonacci"
            f_n = "1"
            
        else:
            a = patron_recursivo["parametros"]["total_llamadas"]
        
        if patron_recursivo["tipo"] not in ["recursion_lineal", "recursion_exponencial_fibonacci"]:
            division_info = patron_recursivo["division"]
            if division_info["tipo"] == "mitad":
                b = 2
            elif division_info["tipo"] == "tercio":
                b = 3
            elif division_info["tipo"] == "n_minus_1":
                b = 1
                tipo_especial = "n_minus_1"
            else:
                b = 2
        
        f_n = self._detectar_costo_no_recursivo_mejorado(ast_obj, patron_recursivo["tipo"])
        
        if tipo_especial == "n_minus_1":
            ecuacion = f"T(n) = T(n-1) + O({f_n})"
        elif tipo_especial == "fibonacci":
            ecuacion = f"T(n) = T(n-1) + T(n-2) + O({f_n})"
        else:
            ecuacion = f"T(n) = {a}T(n/{b}) + O({f_n})"
        
        return {
            "ecuacion": ecuacion,
            "a": a,
            "b": b,
            "f_n": f_n,
            "tipo_especial": tipo_especial,
            "patron": patron_recursivo
        }

    def _detectar_costo_no_recursivo_mejorado(self, ast_obj: 'AST', tipo_recursion: str) -> str:
        """
        Detecta el costo del trabajo no recursivo.
        """
        if tipo_recursion == "recursion_binaria":
            if self._buscar_evidencia_operacion_lineal(ast_obj):
                return "n"
            else:
                return "1"
        
        analisis_estructural = self._analizar_estructura_ast(ast_obj)
        profundidad = analisis_estructural["max_profundidad"]
        
        if profundidad > 1:
            return f"n^{profundidad}"
        elif profundidad == 1:
            return "n"
        else:
            return "1"

    def _buscar_evidencia_operacion_lineal(self, ast_obj: 'AST') -> bool:
        """
        Busca evidencia de operaciones O(n) como 'merge'.
        """
        codigo_completo = ast.unparse(ast_obj._arbol).lower()
        
        palabras_clave_lineales = ["merge", "combine", "fusion", "join", "concatenate"]
        if any(palabra in codigo_completo for palabra in palabras_clave_lineales):
            return True
        
        for node in ast.walk(ast_obj._arbol):
            if isinstance(node, ast.For):
                return True
        
        return False

    def _resolver_recurrencia_con_llm(self, ecuacion: str, patron: Dict[str, Any], pseudocodigo: str) -> Dict[str, Any]:
        """
        Usa LLM para resolver la ecuación de recurrencia de manera precisa.
        """
        prompt = f"""
Eres un experto en análisis de algoritmos del libro "Introduction to Algorithms" (Cormen et al.).

**Pseudocódigo a analizar:**
```pseudocode
{pseudocodigo}
```

Patrón detectado automáticamente:
- Tipo de recursión: {patron['tipo']}
- División del problema: {patron['division']['tipo']}
- Factor de división: {patron['division']['factor']}

Ecuación de recurrencia generada:
{ecuacion}

Instrucciones:
1. Resuelve la ecuación de recurrencia usando los métodos apropiados (Teorema Maestro, sustitución, árbol de recursión)
2. Proporciona las cotas asintóticas EXACTAS (O, Ω, Θ) basadas en el análisis formal
3. Incluye referencias específicas a "Introduction to Algorithms" (capítulo, sección, página)
4. Para Fibonacci, usa la notación exacta Θ(φⁿ) donde φ es la razón áurea
5. Justifica matemáticamente cada paso

Formato de respuesta JSON:
{{
    "notacion_o": "O(...)",
    "notacion_omega": "Ω(...)",
    "notacion_theta": "Θ(...)",
    "justificacion_matematica": "Explicación detallada...",
    "referencias": "Capítulo X, Sección Y, página Z",
    "metodo_utilizado": "Teorema Maestro/Método de Sustitución/Árbol de Recursión"
}}

Responde ÚNICAMENTE con el JSON válido, sin texto adicional.
"""
            
        try:
            respuesta = self._llm_service.analizar_complejidad(prompt)
            respuesta_limpia = self._limpiar_respuesta_json(respuesta)
            return json.loads(respuesta_limpia)
        except Exception as e:
            print(f"❌ Error al resolver recurrencia con LLM: {e}")
            return self._resolver_recurrencia_local(ecuacion, patron)

    def _limpiar_respuesta_json(self, respuesta: str) -> str:
        """
        Limpia la respuesta del LLM para extraer solo el JSON.
        """
        inicio = respuesta.find('{')
        fin = respuesta.rfind('}') + 1

        if inicio != -1 and fin != -1:
            return respuesta[inicio:fin]

        return respuesta.strip()

    def _resolver_recurrencia_local(self, ecuacion: str, patron: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback: Resuelve la recurrencia con lógica local cuando el LLM falla.
        """
        tipo_recursion = patron["tipo"]

        if tipo_recursion == "recursion_exponencial_fibonacci":
            return {
                "notacion_o": "O(2^n)",
                "notacion_omega": "Ω(φ^n)",
                "notacion_theta": "Θ(φ^n)",
                "justificacion_matematica": "Ecuación de Fibonacci: T(n) = T(n-1) + T(n-2) + O(1). Solución exacta: Θ(φ^n) donde φ ≈ 1.618. O(2^n) es cota superior conservadora.",
                "referencias": "Capítulo 27, Sección 27.1",
                "metodo_utilizado": "Ecuación Característica"
            }
        elif tipo_recursion == "recursion_binaria":
            return {
                "notacion_o": "O(n log n)",
                "notacion_omega": "Ω(n log n)", 
                "notacion_theta": "Θ(n log n)",
                "justificacion_matematica": "T(n) = 2T(n/2) + O(n). Caso 2 del Teorema Maestro: f(n) = Θ(n^(log₂2)) = Θ(n), por tanto T(n) = Θ(n log n).",
                "referencias": "Capítulo 4, Sección 4.5",
                "metodo_utilizado": "Teorema Maestro"
            }
        elif tipo_recursion == "recursion_lineal":
            return {
                "notacion_o": "O(n)",
                "notacion_omega": "Ω(n)",
                "notacion_theta": "Θ(n)", 
                "justificacion_matematica": "T(n) = T(n-1) + O(1). Expansión: T(n) = T(0) + n*c = Θ(n).",
                "referencias": "Capítulo 4, Sección 4.3",
                "metodo_utilizado": "Método de Sustitución"
            }
        else:
            return {
                "notacion_o": "O(n)",
                "notacion_omega": "Ω(1)", 
                "notacion_theta": "Complejidad variable",
                "justificacion_matematica": f"Análisis de recurrencia: {ecuacion}. Se requiere análisis específico para determinar cotas exactas.",
                "referencias": "Capítulo 4",
                "metodo_utilizado": "Análisis General"
            }

    def _combinar_analisis(self, analisis_recurrencia: Dict[str, Any], solucion_llm: Dict[str, Any], algoritmo_id: int) -> Complejidad:
        """
        Combina el análisis estructural con la solución matemática del LLM.
        """
        notacion_o = solucion_llm.get('notacion_o', 'O(n)')
        notacion_omega = solucion_llm.get('notacion_omega', 'Ω(1)')
        notacion_theta = solucion_llm.get('notacion_theta', 'Θ(n)')

        justificacion_combinada = (
            f"**ANÁLISIS DE ALGORITMO RECURSIVO**\n\n"
            f"**Tipo de Recursión Detectada:** {analisis_recurrencia['patron']['tipo'].replace('_', ' ').title()}\n"
            f"**Ecuación de Recurrencia:** {analisis_recurrencia['ecuacion']}\n"
            f"**Método de Análisis:** {solucion_llm.get('metodo_utilizado', 'Análisis Matemático')}\n\n"
            f"**Análisis Estructural Automático:**\n"
            f"- Subproblemas (a): {analisis_recurrencia['a']}\n"
            f"- Factor de división (b): {analisis_recurrencia['b']}\n"
            f"- Trabajo no recursivo: O({analisis_recurrencia['f_n']})\n\n"
            f"**Resolución Matemática:**\n{solucion_llm['justificacion_matematica']}\n\n"
            f"**Referencia:** {solucion_llm['referencias']} - Introduction to Algorithms"
        )

        return Complejidad(
            id=algoritmo_id,
            notacion_o=notacion_o,
            notacion_omega=notacion_omega,
            notacion_theta=notacion_theta,
            justificacion=justificacion_combinada
        )

    def analizar_recursivo(self, algoritmo: Algoritmo) -> Complejidad:
        """
        Analiza algoritmos RECURSIVOS usando LLM para resolver ecuaciones de recurrencia.
        """
        if not algoritmo.arbol_sintactico:
            raise ValueError("El algoritmo no tiene un AST. Ejecute el parser primero.")
            
        if not self._detectar_recursividad(algoritmo.arbol_sintactico):
            raise ValueError("El algoritmo no parece ser recursivo. Use 'analizar' para iterativos.")

        # 1. Generar ecuación de recurrencia
        analisis_recurrencia = self._generar_ecuacion_recurrencia_mejorada(algoritmo.arbol_sintactico)
        
        # 2. Usar LLM para resolver la ecuación
        solucion_llm = self._resolver_recurrencia_con_llm(
            analisis_recurrencia["ecuacion"],
            analisis_recurrencia["patron"],
            algoritmo.codigo_fuente
        )
        
        # 3. Combinar resultados
        return self._combinar_analisis(analisis_recurrencia, solucion_llm, algoritmo.id)
