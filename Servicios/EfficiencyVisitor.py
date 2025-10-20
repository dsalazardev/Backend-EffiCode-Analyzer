from __future__ import annotations
import ast
from typing import List, Tuple, Dict

try:
    import sympy
except ImportError:
    print("Dependencia no encontrada. Por favor, instale 'sympy' usando: pip install sympy")
    sympy = None


class EfficiencyVisitor(ast.NodeVisitor):
    """
    Recorre el AST para construir funciones de eficiencia simbólicas T(n)
    para el mejor y peor caso, utilizando sympy para el cálculo.
    """

    def __init__(self):
        if not sympy:
            raise RuntimeError("La librería 'sympy' es necesaria para el análisis de eficiencia.")

        # Símbolos base para el análisis
        self.n = sympy.Symbol('n')
        self.const_idx = 1

        # Costos y desglose por línea
        self.line_costs: List[Tuple[int, str, str]] = []
        self.worst_case_cost = sympy.Integer(0)
        self.best_case_cost = sympy.Integer(0)

    def _get_const(self) -> sympy.Symbol:
        """Genera una nueva constante simbólica (c_1, c_2, ...)."""
        const = sympy.Symbol(f'c_{self.const_idx}')
        self.const_idx += 1
        return const

    def _add_cost(self, node: ast.AST, description: str, worst_cost: sympy.Expr, best_cost: sympy.Expr = None):
        """Método unificado para añadir costos y registrar el desglose."""
        if best_cost is None:
            best_cost = worst_cost

        self.worst_case_cost += worst_cost
        self.best_case_cost += best_cost

        cost_str = f"Peor: {worst_cost}"
        if worst_cost != best_cost:
            cost_str += f", Mejor: {best_cost}"

        self.line_costs.append((node.lineno, cost_str, description))

    def visit_Assign(self, node: ast.Assign):
        """Visita una asignación. Costo constante en ambos casos."""
        costo = self._get_const()
        self._add_cost(node, "Asignación", costo)
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        """Visita un condicional. Analiza ambas ramas para mejor/peor caso."""
        # Costo de evaluar la condición
        costo_test = self._get_const()
        self._add_cost(node, "Evaluación de condición if", costo_test)

        # Analiza la rama 'if'
        visitor_if_body = EfficiencyVisitor()
        for sub_node in node.body:
            visitor_if_body.visit(sub_node)

        # Analiza la rama 'else' (si existe)
        visitor_else_body = EfficiencyVisitor()
        if node.orelse:
            for sub_node in node.orelse:
                visitor_else_body.visit(sub_node)

        # El peor caso es el de la rama más costosa, el mejor, el de la menos costosa
        peor_rama = sympy.Max(visitor_if_body.worst_case_cost, visitor_else_body.worst_case_cost)
        mejor_rama = sympy.Min(visitor_if_body.best_case_cost, visitor_else_body.best_case_cost)

        self.worst_case_cost += peor_rama
        self.best_case_cost += mejor_rama

        # Agrega los desgloses de ambas ramas para un reporte completo
        self.line_costs.extend([(ln, c, f"  (Rama if) {d}") for ln, c, d in visitor_if_body.line_costs])
        if node.orelse:
            self.line_costs.extend([(ln, c, f"  (Rama else) {d}") for ln, c, d in visitor_else_body.line_costs])

    def visit_For(self, node: ast.For):
        """Visita un bucle 'for'. Costo es una sumatoria."""
        iter_var = sympy.Symbol(node.target.id)
        # Simplificación: asumimos que itera n veces (de 1 a n)
        limite_superior = self.n

        # Costo de inicialización y test del bucle
        self._add_cost(node, f"Inicialización y test del bucle for", self._get_const())

        # Analiza el cuerpo del bucle
        visitor_cuerpo = EfficiencyVisitor()
        for sub_node in node.body:
            visitor_cuerpo.visit(sub_node)

        # La sumatoria se aplica a ambos casos (peor y mejor)
        sumatoria_peor = sympy.Sum(visitor_cuerpo.worst_case_cost, (iter_var, 1, limite_superior))
        sumatoria_mejor = sympy.Sum(visitor_cuerpo.best_case_cost, (iter_var, 1, limite_superior))

        self.worst_case_cost += sumatoria_peor
        self.best_case_cost += sumatoria_mejor
        self.line_costs.extend([(ln, c, f"  (Dentro de for) {d}") for ln, c, d in visitor_cuerpo.line_costs])

    def visit_While(self, node: ast.While):
        """Visita un bucle 'while'."""
        # t_j es el número de veces que el cuerpo del while se ejecuta
        # En el peor caso, es 'n'. En el mejor caso, podría ser 1 si hay un break.
        tj_peor = self.n
        tj_mejor = sympy.Symbol('t_j_mejor', positive=True, integer=True)  # Lo dejamos simbólico

        # Analiza el cuerpo del bucle
        visitor_cuerpo = EfficiencyVisitor()
        hay_break = any(isinstance(sn, ast.Break) for sn in ast.walk(node))
        if hay_break:
            # Si hay un break, el mejor caso es que el bucle se ejecute 1 vez.
            tj_mejor = 1

        for sub_node in node.body:
            visitor_cuerpo.visit(sub_node)

        # El test se ejecuta t_j + 1 veces
        costo_test = self._get_const()
        self._add_cost(node, f"Test de condición while",
                       worst_cost=sympy.Sum(costo_test, (sympy.Symbol('i'), 1, tj_peor + 1)),
                       best_cost=sympy.Sum(costo_test, (sympy.Symbol('i'), 1, tj_mejor + 1)))

        # El cuerpo se ejecuta t_j veces
        self.worst_case_cost += sympy.Sum(visitor_cuerpo.worst_case_cost, (sympy.Symbol('j'), 1, tj_peor))
        self.best_case_cost += sympy.Sum(visitor_cuerpo.best_case_cost, (sympy.Symbol('j'), 1, tj_mejor))
        self.line_costs.extend([(ln, c, f"  (Dentro de while) {d}") for ln, c, d in visitor_cuerpo.line_costs])