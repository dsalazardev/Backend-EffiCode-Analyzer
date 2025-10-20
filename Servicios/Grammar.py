from __future__ import annotations
from typing import Dict, Any

# Se recomienda instalar lark: pip install lark-parser
try:
    from lark import Lark, exceptions
except ImportError:
    print("Dependencia no encontrada. Por favor, instale 'lark-parser' usando: pip install lark-parser")
    Lark = object


class Grammar:
    """
    Define y valida la sintaxis del pseudocódigo inspirado en el libro
    "Introduction to Algorithms" (Cormen et al.).

    Esta clase utiliza la librería Lark para construir un parser basado en una
    gramática EBNF formal. Permite validar que el pseudocódigo de entrada
    sea sintácticamente correcto antes de proceder a su análisis.
    """

    # Gramática EBNF que define la sintaxis del pseudocódigo estilo Cormen.
    # Incluye el operador de asignación '←' y los relacionales '≤, ≥, ≠'.
    _pseudocode_grammar = r"""
        ?start: (sentencia | declaracion_funcion)+

        ?sentencia: asignacion
                  | if_sentencia
                  | for_sentencia
                  | while_sentencia
                  | call_funcion
                  | return_sentencia

        declaracion_funcion: IDENTIFICADOR "(" [parametro_lista] ")" sentencias

        if_sentencia: "if" condicion "then" sentencias ("else" sentencias)?
        for_sentencia: "for" IDENTIFICADOR "←" expresion ("to" | "downto") expresion "do" sentencias
        while_sentencia: "while" condicion "do" sentencias

        // Un bloque de sentencias es simplemente una o más sentencias.
        // La indentación de Cormen se modela permitiendo secuencias de sentencias.
        sentencias: (sentencia)+

        asignacion: variable "←" expresion
        return_sentencia: "return" expresion

        parametro_lista: IDENTIFICADOR ("," IDENTIFICADOR)*
        argumento_lista: expresion ("," expresion)*

        ?condicion: expresion

        ?expresion: bool_or
        ?bool_or: bool_and ("or" bool_and)*
        ?bool_and: bool_not ("and" bool_not)*
        ?bool_not: "not" bool_not | comparacion
        ?comparacion: aritmetica (REL_OP aritmetica)*

        ?aritmetica: term (("+"|"-") term)*
        ?term: factor (("*"|"/"|"div"|"mod") factor)*
        ?factor: ("+"|"-") factor | atomo

        ?atomo: NUMBER
             | variable
             | "(" expresion ")"
             | call_funcion

        variable: IDENTIFICADOR ("." IDENTIFICADOR | "[" expresion "]")*
        call_funcion: IDENTIFICADOR "(" [argumento_lista] ")"

        // Un identificador puede contener letras, números, guiones bajos y guiones medios.
        IDENTIFICADOR: /[a-zA-Z_][a-zA-Z0-9_-]*/
        REL_OP: "≤" | "≥" | "≠" | "=" | "<" | ">" //  ≤ (\u2264), ≥ (\u2265), ≠ (\u2260)

        COMMENT: "//" /[^\n]*/

        %import common.NUMBER
        %import common.WS
        %ignore WS
        %ignore COMMENT
    """

    def __init__(self):
        if Lark is not object:
            self.parser = Lark(self._pseudocode_grammar, start='start', parser='earley')
        else:
            self.parser = None
        self._reglas = self._parse_grammar_to_dict()

    def _parse_grammar_to_dict(self) -> Dict[str, str]:
        reglas_dict = {}
        for line in self._pseudocode_grammar.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if ":" in line:
                nombre, definicion = line.split(":", 1)
                reglas_dict[nombre.strip().replace('?', '')] = definicion.strip()
        return reglas_dict

    def obtener_regla(self, nombre: str) -> str | None:
        return self._reglas.get(nombre)

    def listar_reglas(self) -> Dict[str, str]:
        return self._reglas

    def validar_sentencia(self, codigo: str) -> bool:

        if self.parser is None:
            print("Lark no está instalado. No se puede validar la sintaxis.")
            return False

        try:
            self.parser.parse(codigo)
            return True
        except exceptions.LarkError as e:
            print(f"Error de sintaxis: {e}")
            return False


