from __future__ import annotations
import ast
from typing import List, Tuple, Dict, Optional

try:
    import sympy
    from sympy import Symbol, Sum, Integer, Max, Min, simplify
except ImportError:
    print("Dependencia no encontrada. Por favor, instale 'sympy' usando: pip install sympy")
    sympy = None


class EfficiencyVisitor(ast.NodeVisitor):
    """
    Recorre el AST para construir funciones de eficiencia simbólicas T(n)
    para el mejor y peor caso, utilizando sympy para el cálculo.
    
    Genera sumatorias anidadas correctas para bucles anidados estilo Cormen.
    """

    def __init__(self, loop_context: Optional[List[Tuple[Symbol, Symbol, Symbol]]] = None):
        """
        Args:
            loop_context: Lista de tuplas (variable, inicio, fin) que representan
                         los bucles que contienen este contexto. Usado para construir
                         sumatorias anidadas.
        """
        if not sympy:
            raise RuntimeError("La librería 'sympy' es necesaria para el análisis de eficiencia.")

        # Símbolos base para el análisis
        self.n = Symbol('n', positive=True, integer=True)
        self.const_idx = 1

        # Contexto de bucles (para sumatorias anidadas)
        self.loop_context = loop_context or []

        # Costos y desglose por línea: (lineno, worst_expr, best_expr, description)
        self.line_costs: List[Tuple[int, 'sympy.Expr', 'sympy.Expr', str]] = []
        self.worst_case_cost = Integer(0)
        self.best_case_cost = Integer(0)
        
        # Contador global de constantes (para mantener consistencia)
        self._global_const_counter = 1

    def _get_const(self) -> Symbol:
        """Genera una nueva constante simbólica (c_1, c_2, ...)."""
        const = Symbol(f'c_{self.const_idx}', positive=True)
        self.const_idx += 1
        return const

    def _wrap_in_sums(self, expr: sympy.Expr) -> sympy.Expr:
        """
        Envuelve una expresión en las sumatorias correspondientes
        al contexto de bucles actual.
        
        Por ejemplo, si estamos dentro de:
            for i = 1 to n
                for j = i to n
        
        El costo c_k se convierte en: Σ(i=1 to n) Σ(j=i to n) c_k
        """
        result = expr
        # Aplicar sumatorias de adentro hacia afuera
        for var, start, end in reversed(self.loop_context):
            result = Sum(result, (var, start, end))
        return result

    def _get_loop_description(self) -> str:
        """Genera una descripción del contexto de bucles actual."""
        if not self.loop_context:
            return ""
        
        depth = len(self.loop_context)
        prefix = "  " * depth
        return f"{prefix}(Dentro de {'for' if depth == 1 else 'bucle anidado'}) "

    def _add_cost(self, node: ast.AST, description: str, worst_cost: sympy.Expr, best_cost: sympy.Expr = None):
        """Método unificado para añadir costos y registrar el desglose."""
        if best_cost is None:
            best_cost = worst_cost

        # Envolver en sumatorias según el contexto
        wrapped_worst = self._wrap_in_sums(worst_cost)
        wrapped_best = self._wrap_in_sums(best_cost)

        self.worst_case_cost += wrapped_worst
        self.best_case_cost += wrapped_best

        # Construir descripción con contexto de bucles
        depth_prefix = "  " * len(self.loop_context)
        loop_desc = ""
        for i, (var, start, end) in enumerate(self.loop_context):
            loop_desc += f"(Dentro de for {var}) "
        
        full_description = f"{depth_prefix}{loop_desc}{description}"

        # Guardar las expresiones sympy con sumatorias para render LaTeX
        self.line_costs.append((node.lineno, wrapped_worst, wrapped_best, full_description))

    def visit_Assign(self, node: ast.Assign):
        """Visita una asignación. Costo constante multiplicado por sumatorias de bucles."""
        costo = self._get_const()
        self._add_cost(node, "Asignación", costo)
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        """Visita un condicional. Analiza ambas ramas para mejor/peor caso."""
        # Costo de evaluar la condición
        costo_test = self._get_const()
        self._add_cost(node, "Evaluación de condición if", costo_test)

        # Analiza la rama 'if' con el mismo contexto de bucles
        visitor_if_body = EfficiencyVisitor(loop_context=self.loop_context.copy())
        visitor_if_body.const_idx = self.const_idx
        for sub_node in node.body:
            visitor_if_body.visit(sub_node)
        self.const_idx = visitor_if_body.const_idx

        # Analiza la rama 'else' (si existe)
        visitor_else_body = EfficiencyVisitor(loop_context=self.loop_context.copy())
        visitor_else_body.const_idx = self.const_idx
        if node.orelse:
            for sub_node in node.orelse:
                visitor_else_body.visit(sub_node)
            self.const_idx = visitor_else_body.const_idx

        # El peor caso es el de la rama más costosa, el mejor, el de la menos costosa
        peor_rama = Max(visitor_if_body.worst_case_cost, visitor_else_body.worst_case_cost)
        mejor_rama = Min(visitor_if_body.best_case_cost, visitor_else_body.best_case_cost)

        self.worst_case_cost += peor_rama
        self.best_case_cost += mejor_rama

        # Agrega los desgloses de ambas ramas
        depth_prefix = "  " * (len(self.loop_context) + 1)
        self.line_costs.extend([
            (ln, worst, best, f"{depth_prefix}(Rama if) {d}") 
            for ln, worst, best, d in visitor_if_body.line_costs
        ])
        if node.orelse:
            self.line_costs.extend([
                (ln, worst, best, f"{depth_prefix}(Rama else) {d}") 
                for ln, worst, best, d in visitor_else_body.line_costs
            ])

    def visit_For(self, node: ast.For):
        """
        Visita un bucle 'for'. 
        Crea una sumatoria que envuelve todos los costos del cuerpo.
        """
        # Obtener variable del bucle
        if isinstance(node.target, ast.Name):
            iter_var = Symbol(node.target.id, positive=True, integer=True)
        else:
            iter_var = Symbol('i', positive=True, integer=True)

        # Intentar extraer límites del range()
        start_expr = Integer(1)
        end_expr = self.n
        is_downto = False
        
        if isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
            if node.iter.func.id == 'range':
                args = node.iter.args
                if len(args) == 2:
                    # range(start, end)
                    start_expr = self._ast_to_sympy(args[0])
                    end_expr = self._ast_to_sympy(args[1]) - 1  # range es exclusivo
                elif len(args) == 3:
                    # range(start, end, step) - para downto
                    start_expr = self._ast_to_sympy(args[0])
                    end_expr = self._ast_to_sympy(args[1]) + 1  # ajuste para downto
                    is_downto = True

        # Registrar costo de inicialización del bucle
        init_cost = self._get_const()
        self._add_cost(node, f"Inicialización y test del bucle for ({iter_var})", init_cost)

        # Crear nuevo contexto de bucles para el cuerpo
        new_loop_context = self.loop_context.copy()
        if is_downto:
            new_loop_context.append((iter_var, end_expr, start_expr))
        else:
            new_loop_context.append((iter_var, start_expr, end_expr))

        # Analizar el cuerpo del bucle con el nuevo contexto
        visitor_cuerpo = EfficiencyVisitor(loop_context=new_loop_context)
        visitor_cuerpo.const_idx = self.const_idx
        for sub_node in node.body:
            visitor_cuerpo.visit(sub_node)
        self.const_idx = visitor_cuerpo.const_idx

        # Agregar los costos del cuerpo (ya vienen con las sumatorias correctas)
        self.worst_case_cost += visitor_cuerpo.worst_case_cost
        self.best_case_cost += visitor_cuerpo.best_case_cost
        self.line_costs.extend(visitor_cuerpo.line_costs)

    def visit_While(self, node: ast.While):
        """Visita un bucle 'while'."""
        # Para while, usamos t_j como número de iteraciones
        j = Symbol('j', positive=True, integer=True)
        tj_peor = self.n
        tj_mejor = Integer(1)

        # Verificar si hay break (mejor caso = 1 iteración)
        hay_break = any(isinstance(sn, ast.Break) for sn in ast.walk(node))
        if not hay_break:
            tj_mejor = self.n

        # Costo del test de condición
        costo_test = self._get_const()
        self._add_cost(node, "Test de condición while", costo_test)

        # Crear contexto para el cuerpo del while
        new_loop_context = self.loop_context.copy()
        new_loop_context.append((j, Integer(1), tj_peor))

        # Analizar el cuerpo
        visitor_cuerpo = EfficiencyVisitor(loop_context=new_loop_context)
        visitor_cuerpo.const_idx = self.const_idx
        for sub_node in node.body:
            visitor_cuerpo.visit(sub_node)
        self.const_idx = visitor_cuerpo.const_idx

        self.worst_case_cost += visitor_cuerpo.worst_case_cost
        self.best_case_cost += visitor_cuerpo.best_case_cost
        self.line_costs.extend(visitor_cuerpo.line_costs)

    def _ast_to_sympy(self, node: ast.AST) -> sympy.Expr:
        """Convierte un nodo AST a una expresión sympy."""
        if isinstance(node, ast.Constant):
            return Integer(node.value)
        elif isinstance(node, ast.Name):
            return Symbol(node.id, positive=True, integer=True)
        elif isinstance(node, ast.BinOp):
            left = self._ast_to_sympy(node.left)
            right = self._ast_to_sympy(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                return left / right
        elif isinstance(node, ast.UnaryOp):
            operand = self._ast_to_sympy(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
        # Default: devolver n
        return self.n