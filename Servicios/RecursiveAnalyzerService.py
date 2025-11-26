"""
Servicio para análisis de complejidad de algoritmos RECURSIVOS.
Detecta patrones de recursión y genera ecuaciones de recurrencia.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any
import ast
import json

if TYPE_CHECKING:
    from Servicios.Ast import AST
    from Servicios.LLMService import LLMService


class RecursiveAnalyzerService:
    """
    Servicio especializado en análisis de algoritmos recursivos.
    Detecta patrones de recursión, genera ecuaciones de recurrencia
    y utiliza LLM para resolver las ecuaciones.
    """

    def __init__(self, llm_service: 'LLMService' = None):
        self._llm_service = llm_service

    @property
    def llm_service(self) -> 'LLMService':
        return self._llm_service

    @llm_service.setter
    def llm_service(self, value: 'LLMService'):
        self._llm_service = value

    def detectar_recursividad(self, ast_obj: 'AST') -> bool:
        """
        Detecta si la función principal realiza una llamada recursiva a sí misma.
        """
        funciones_definidas = ast_obj.extraer_funciones()
        if not funciones_definidas:
            return False

        nombre_funcion_principal = funciones_definidas[0]
        llamadas = ast_obj.extraer_llamadas()
        
        return nombre_funcion_principal in llamadas

    def detectar_patron_recursivo(self, ast_obj: 'AST') -> Dict[str, Any]:
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

    def generar_ecuacion_recurrencia(self, ast_obj: 'AST') -> Dict[str, Any]:
        """
        Genera la ecuación de recurrencia basada en el análisis estructural.
        """
        patron_recursivo = self.detectar_patron_recursivo(ast_obj)
        
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
        
        f_n = self._detectar_costo_no_recursivo(ast_obj, patron_recursivo["tipo"])
        
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

    def _detectar_costo_no_recursivo(self, ast_obj: 'AST', tipo_recursion: str) -> str:
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

    def _analizar_estructura_ast(self, ast_obj: 'AST') -> Dict[str, Any]:
        """
        Analiza la estructura del AST para determinar profundidad de bucles.
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

    def resolver_recurrencia_con_llm(self, ecuacion: str, patron: Dict[str, Any], pseudocodigo: str) -> Dict[str, Any]:
        """
        Usa LLM para resolver la ecuación de recurrencia de manera precisa.
        """
        if not self._llm_service:
            return self.resolver_recurrencia_local(ecuacion, patron)
        
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
            return self.resolver_recurrencia_local(ecuacion, patron)

    def _limpiar_respuesta_json(self, respuesta: str) -> str:
        """
        Limpia la respuesta del LLM para extraer solo el JSON.
        """
        inicio = respuesta.find('{')
        fin = respuesta.rfind('}') + 1

        if inicio != -1 and fin != -1:
            return respuesta[inicio:fin]

        return respuesta.strip()

    def resolver_recurrencia_local(self, ecuacion: str, patron: Dict[str, Any]) -> Dict[str, Any]:
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

    def generar_justificacion_combinada(self, analisis_recurrencia: Dict[str, Any], solucion_llm: Dict[str, Any]) -> str:
        """
        Genera la justificación combinada para el análisis recursivo.
        """
        return (
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
